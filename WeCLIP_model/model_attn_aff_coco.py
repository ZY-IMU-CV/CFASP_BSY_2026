import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize

import clip
from clip.clip_text import BACKGROUND_CATEGORY_COCO, new_class_names_coco
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_coco_cam
from pytorch_grad_cam import GradCAM
from WeCLIP_model.PAR import PAR

from .Decoder.TransDecoder import DecoderTransformer
from .segformer_head import SegFormerHead


def Normalize_clip():
    return Compose([
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
    ])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(
        tensor.size(0), height, width, tensor.size(2)
    )
    return result.transpose(2, 3).transpose(1, 2)


def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights.t()


def _refine_cams(ref_mod, images, cams, valid_key):
    images = images.unsqueeze(0)
    cams = cams.unsqueeze(0)
    refined_cams = ref_mod(images.float(), cams.float())
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]
    return refined_label.squeeze(0)


class WeCLIP(nn.Module):
    """COCO WeCLIP model with the same CLIP+DINO path used by the VOC model."""

    def __init__(
        self,
        num_classes=None,
        clip_model=None,
        embedding_dim=256,
        in_channels=512,
        dataset_root_path=None,
        device="cuda",
        use_segformer_adapter=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_segformer_adapter = use_segformer_adapter
        self.dino_model_name = "dinov2_vitb14"
        self.dino_fts_fuse_dim = 768
        self.dino_fusion_weight = 0.5

        self.encoder, _ = clip.load(clip_model, device=device)
        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad = False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        try:
            self.dino_encoder = torch.hub.load(
                "facebookresearch/dinov2",
                self.dino_model_name,
                force_reload=False,
                verbose=False,
            )
        except Exception:
            local_hub_dir = os.path.expanduser(
                "~/.cache/torch/hub/facebookresearch_dinov2_main"
            )
            self.dino_encoder = torch.hub.load(
                local_hub_dir,
                self.dino_model_name,
                source="local",
                force_reload=False,
            )

        for param in self.dino_encoder.parameters():
            param.requires_grad = False

        self.in_channels = in_channels
        self.decoder_fts_fuse = SegFormerHead(
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            index=1,
            use_adapter=self.use_segformer_adapter,
        )
        self.dino_decoder_fts_fuse = SegFormerHead(
            in_channels=[self.dino_fts_fuse_dim] * 4,
            embedding_dim=self.embedding_dim,
            num_classes=self.num_classes,
            index=1,
            use_adapter=self.use_segformer_adapter,
        )
        self.decoder = DecoderTransformer(
            width=self.embedding_dim,
            layers=3,
            heads=8,
            output_dim=self.num_classes,
        )

        self.bg_text_features = zeroshot_classifier(
            BACKGROUND_CATEGORY_COCO, ["a clean origami {}."], self.encoder
        )
        self.fg_text_features = zeroshot_classifier(
            new_class_names_coco, ["a clean origami {}."], self.encoder
        )

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(
            model=self.encoder,
            target_layers=self.target_layers,
            reshape_transform=reshape_transform,
        )
        self.root_path = os.path.join(dataset_root_path, "SegmentationClass")
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1, 2, 4, 8, 12, 24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True

    def get_param_groups(self):
        param_groups = [[], [], [], []]
        param_groups[3].extend(self.decoder.parameters())
        param_groups[3].extend(self.decoder_fts_fuse.parameters())
        param_groups[3].extend(self.dino_decoder_fts_fuse.parameters())
        return param_groups

    def _extract_dino_features(self, img, target_size):
        b, _, h, w = img.shape
        self.dino_encoder.eval()
        dino_img_h = max(14, (h // 14) * 14)
        dino_img_w = max(14, (w // 14) * 14)
        dino_img = F.interpolate(
            img,
            size=(dino_img_h, dino_img_w),
            mode="bilinear",
            align_corners=False,
        )

        with torch.no_grad():
            dino_outputs = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_outputs["x_norm_patchtokens"]

        if isinstance(dino_fts, list):
            dino_fts = [
                feature.reshape(
                    b, dino_img_h // 14, dino_img_w // 14, -1
                ).permute(0, 3, 1, 2)
                for feature in dino_fts
            ]
            dino_fts = self.dino_decoder_fts_fuse(torch.stack(dino_fts))
        else:
            dino_fts = dino_fts.reshape(
                b, dino_img_h // 14, dino_img_w // 14, -1
            ).permute(0, 3, 1, 2)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

        return F.interpolate(
            dino_fts, size=target_size, mode="bilinear", align_corners=False
        )

    def forward(
        self,
        img,
        img_names,
        mode="train",
        cls_labels=None,
        return_refined_cams=False,
    ):
        cam_list = []
        refined_cam_infos = []
        b, _, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(
            img, self.encoder, require_all_fts=True
        )
        fts_all_stack = torch.stack(fts_all, dim=0)
        attn_weight_stack = torch.stack(attn_weight_list, dim=0).permute(
            1, 0, 2, 3
        )
        if self.require_all_fts:
            cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3)
        else:
            cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(
            -1, b, img_tokens_channel, h // 16, w // 16
        )
        all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape
        dino_fts = self._extract_dino_features(img, target_size=fts.shape[-2:])

        seg_clip, _ = self.decoder(fts)
        seg_dino, _ = self.decoder(dino_fts)
        seg = (
            (1.0 - self.dino_fusion_weight) * seg_clip
            + self.dino_fusion_weight * seg_dino
        )

        seg_dino_prob = F.softmax(seg, dim=1).detach()

        clip_dino_fts = torch.cat([fts, dino_fts], dim=1)
        attn_fts = F.interpolate(
            clip_dino_fts,
            size=(fts_h, fts_w),
            mode="bilinear",
            align_corners=False,
        )
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h * f_w)
        attn_pred = torch.sigmoid(
            attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        )

        split_dir = "val" if mode == "val" else "train"
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(
                self.root_path, split_dir, str(img_name) + ".png"
            )
            label_id_list = None
            if cls_labels is not None:
                sample_cls = cls_labels[i]
                if torch.is_tensor(sample_cls):
                    label_id_list = (
                        torch.nonzero(sample_cls > 0, as_tuple=False)
                        .view(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                else:
                    label_id_list = np.where(np.asarray(sample_cls) > 0)[0].tolist()

            cam_refined_list, keys, cam_w, cam_h = perform_single_coco_cam(
                img_path,
                img[i],
                cam_fts_all[i],
                attn_weight_stack[i],
                attn_pred.unsqueeze(0)[:, i, :, :],
                self.bg_text_features,
                self.fg_text_features,
                self.grad_cam,
                mode=mode,
                require_seg_trans=True,
                label_id_list=label_id_list,
                seg_dino_cam=seg_dino_prob[i],
            )

            cam_dict = generate_cam_label(
                cam_refined_list, keys, cam_w, cam_h
            )
            if return_refined_cams:
                refined_cam_infos.append(
                    {
                        "keys": cam_dict["keys"],
                        "refined_cam": cam_dict["refined_cam"].detach().cpu(),
                    }
                )

            cams = cam_dict["refined_cam"].cuda()
            bg_score = torch.pow(
                1 - torch.max(cams, dim=0, keepdims=True)[0],
                self.cam_bg_thres,
            ).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()

            valid_key = np.pad(cam_dict["keys"] + 1, (1, 0), mode="constant")
            valid_key = torch.from_numpy(valid_key).cuda()
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)
        if mode == "train" and return_refined_cams:
            return (
                seg,
                seg_clip,
                seg_dino,
                all_cam_labels,
                attn_pred,
                refined_cam_infos,
            )
        if mode == "train":
            return seg, seg_clip, seg_dino, all_cam_labels, attn_pred
        if return_refined_cams:
            return seg, all_cam_labels, attn_pred, refined_cam_infos
        return seg, all_cam_labels, attn_pred
