import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead
import numpy as np
import clip
from clip.clip_text import new_class_names, BACKGROUND_CATEGORY
from pytorch_grad_cam import GradCAM
from clip.clip_tool import generate_cam_label, generate_clip_fts, perform_single_voc_cam
import os
from torchvision.transforms import Compose, Normalize
from .Decoder.TransDecoder import DecoderTransformer
from WeCLIP_model.PAR import PAR




def Normalize_clip():
    return Compose([
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])


def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result



def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
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
    def __init__(self, num_classes=None, clip_model=None, embedding_dim=256, in_channels=512, 
                 dataset_root_path=None, device='cuda', use_segformer_adapter=False,
                 use_deca=True):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.use_deca = bool(use_deca)
        # The LEC-style adapter belongs to DECA.  The ablated baseline must use
        # the original CLIP MLP + Transformer decoder instead.
        self.use_segformer_adapter = bool(use_segformer_adapter) and self.use_deca
        self.clip_feature_layers = 1 if self.use_deca else 11
        self.dino_model_name = 'dinov2_vitb14'
        self.dino_fts_fuse_dim = 768
        self.dino_fusion_weight = 0.5

        # CLIP加载
        self.encoder, _ = clip.load(clip_model, device=device)

        # CLIP冻结
        for name, param in self.encoder.named_parameters():
            if "11" not in name:
                param.requires_grad=False

        for name, param in self.encoder.named_parameters():
            print(name, param.requires_grad)

        self.dino_encoder = None
        if self.use_deca:
            # DINOv2 is part of DECA and is not loaded for the single-CLIP
            # ablation, which also avoids its memory and startup overhead.
            try:
                self.dino_encoder = torch.hub.load(
                    'facebookresearch/dinov2',
                    self.dino_model_name,
                    force_reload=False,
                    verbose=False
                )
            except Exception:
                local_hub_dir = os.path.expanduser('~/.cache/torch/hub/facebookresearch_dinov2_main')
                self.dino_encoder = torch.hub.load(
                    local_hub_dir,
                    self.dino_model_name,
                    source='local',
                    force_reload=False
                )

            for param in self.dino_encoder.parameters():
                param.requires_grad = False

        self.in_channels = in_channels

        # MLP
        self.decoder_fts_fuse = SegFormerHead(in_channels=self.in_channels, embedding_dim=self.embedding_dim,
                                              num_classes=self.num_classes, index=self.clip_feature_layers,
                                              use_adapter=self.use_segformer_adapter)  # index=11
        self.dino_decoder_fts_fuse = None
        if self.use_deca:
            self.dino_decoder_fts_fuse = SegFormerHead(in_channels=[self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim, self.dino_fts_fuse_dim], embedding_dim=self.embedding_dim,
                                                  num_classes=self.num_classes, index=1,
                                                  use_adapter=self.use_segformer_adapter)
        # 共享decoder
        self.decoder = DecoderTransformer(width=self.embedding_dim, layers=3, heads=8, output_dim=self.num_classes)

        self.bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY, ['a clean origami {}.'], self.encoder)
        self.fg_text_features = zeroshot_classifier(new_class_names, ['a clean origami {}.'], self.encoder)

        self.target_layers = [self.encoder.visual.transformer.resblocks[-1].ln_1]
        self.grad_cam = GradCAM(model=self.encoder, target_layers=self.target_layers, reshape_transform=reshape_transform)
        self.root_path = os.path.join(dataset_root_path, 'SegmentationClassAug')
        self.cam_bg_thres = 1
        self.encoder.eval()
        self.par = PAR(num_iter=20, dilations=[1,2,4,8,12,24]).cuda()
        self.iter_num = 0
        self.require_all_fts = True


    def get_param_groups(self):

        param_groups = [[], [], [], []]  # backbone; backbone_norm; cls_head; seg_head;

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)
        for param in list(self.decoder_fts_fuse.parameters()):
            param_groups[3].append(param)
        if self.dino_decoder_fts_fuse is not None:
            for param in list(self.dino_decoder_fts_fuse.parameters()):
                param_groups[3].append(param)

        return param_groups


    def _extract_dino_features(self, img, target_size):
        if not self.use_deca or self.dino_encoder is None or self.dino_decoder_fts_fuse is None:
            raise RuntimeError('DINO features were requested while DECA is disabled.')

        b, _, h, w = img.shape
        self.dino_encoder.eval()
        dino_img_h, dino_img_w = (h // 14) * 14, (w // 14) * 14
        dino_img = F.interpolate(img, size=(dino_img_h, dino_img_w), mode='bilinear', align_corners=False)

        with torch.no_grad():
            dino_outputs = self.dino_encoder.forward_features(dino_img)
            dino_fts = dino_outputs['x_norm_patchtokens']

        if isinstance(dino_fts, list):
            for d_i, dino_fts_single in enumerate(dino_fts):
                dino_fts[d_i] = dino_fts_single.reshape(
                    b, dino_img_h // 14, dino_img_w // 14, -1
                ).permute(0, 3, 1, 2)
            dino_fts = torch.stack(dino_fts)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts)
        else:
            dino_fts = dino_fts.reshape(
                b, dino_img_h // 14, dino_img_w // 14, -1
            ).permute(0, 3, 1, 2)
            dino_fts = self.dino_decoder_fts_fuse(dino_fts.unsqueeze(0))

        return F.interpolate(dino_fts, size=target_size, mode='bilinear', align_corners=False)


    def forward(self, img, img_names='2007_000032', mode='train', cls_labels=None,
                return_refined_cams=False, return_visualization=False,
                skip_cam_generation=False):
        if skip_cam_generation and mode == 'train':
            raise ValueError("skip_cam_generation is only supported for inference")
        if skip_cam_generation and (return_refined_cams or return_visualization):
            raise ValueError(
                "CAM details and visualization cannot be returned when "
                "skip_cam_generation=True"
            )
        if return_visualization:
            return_refined_cams = True
        cam_list = []
        refined_cam_infos = []
        visualization_cam_infos = []
        b, c, h, w = img.shape
        self.encoder.eval()
        self.iter_num += 1

        fts_all, attn_weight_list = generate_clip_fts(img, self.encoder, require_all_fts=True)

        fts_all_stack = torch.stack(fts_all, dim=0) # (11, hw, b, c)
        if not skip_cam_generation:
            attn_weight_stack = torch.stack(
                attn_weight_list, dim=0
            ).permute(1, 0, 2, 3)
            if self.require_all_fts==True:
                cam_fts_all = fts_all_stack[-1].unsqueeze(0).permute(2, 1, 0, 3) #(1, hw, 1, c)
            else:
                cam_fts_all = fts_all_stack.permute(2, 1, 0, 3)

        all_img_tokens = fts_all_stack[:, 1:, ...]
        img_tokens_channel = all_img_tokens.size(-1)
        all_img_tokens = all_img_tokens.permute(0, 2, 3, 1)
        all_img_tokens = all_img_tokens.reshape(-1, b, img_tokens_channel, h//16, w //16) #(11, b, c, h, w)
        if self.use_deca:
            all_img_tokens = all_img_tokens[-1].unsqueeze(0)

        fts = self.decoder_fts_fuse(all_img_tokens)
        _, _, fts_h, fts_w = fts.shape  # 24

        seg_clip, _ = self.decoder(fts)
        if self.use_deca:
            dino_fts = self._extract_dino_features(img, target_size=fts.shape[-2:])
            seg_dino, _ = self.decoder(dino_fts)
            seg = (1.0 - self.dino_fusion_weight) * seg_clip + self.dino_fusion_weight * seg_dino
            seg_dino_prob = F.softmax(seg, dim=1).detach()
            attn_fts = torch.cat([fts, dino_fts], dim=1)
        else:
            # Original WeCLIP baseline: the CLIP features still pass through a
            # complete feature-fusion head and Transformer segmentation decoder.
            seg_dino = None
            seg = seg_clip
            seg_dino_prob = None
            attn_fts = fts

        # The official VOC test split has neither pixel GT nor image-level
        # class labels. Decoder logits remain valid, while CAM/affinity
        # construction has no target and can be skipped entirely.
        if skip_cam_generation:
            return seg, None, None

        attn_fts = F.interpolate(attn_fts, size=(fts_h, fts_w), mode='bilinear', align_corners=False)
        f_b, f_c, f_h, f_w = attn_fts.shape
        attn_fts_flatten = attn_fts.reshape(f_b, f_c, f_h * f_w)
        attn_pred = attn_fts_flatten.transpose(2, 1).bmm(attn_fts_flatten)
        attn_pred = torch.sigmoid(attn_pred)

        for i, img_name in enumerate(img_names):
            img_path = os.path.join(self.root_path, str(img_name)+'.png')
            img_i = img[i]
            cam_fts = cam_fts_all[i]
            cam_attn = attn_weight_stack[i]
            seg_attn = attn_pred.unsqueeze(0)[:, i, :, :]
            label_id_list = None
            if cls_labels is not None:
                sample_cls = cls_labels[i]
                if torch.is_tensor(sample_cls):
                    label_id_list = torch.nonzero(sample_cls > 0, as_tuple=False).view(-1).detach().cpu().numpy().tolist()
                else:
                    label_id_list = np.where(np.asarray(sample_cls) > 0)[0].tolist()

            require_seg_trans = True

            cam_outputs = perform_single_voc_cam(
                img_path, img_i, cam_fts, cam_attn, seg_attn,
                self.bg_text_features, self.fg_text_features,
                self.grad_cam,
                mode=mode,
                require_seg_trans=require_seg_trans,
                label_id_list=label_id_list,
                seg_dino_cam=(
                    seg_dino_prob[i]
                    if seg_dino_prob is not None
                    else None
                ),
                return_initial_cams=return_visualization,
            )
            if return_visualization:
                cam_refined_list, keys, w, h, initial_cams = cam_outputs
            else:
                cam_refined_list, keys, w, h = cam_outputs

            cam_dict = generate_cam_label(cam_refined_list, keys, w, h)
            if return_refined_cams:
                refined_cam_infos.append({
                    'keys': cam_dict['keys'],
                    'refined_cam': cam_dict['refined_cam'].detach().cpu()
                })
            if return_visualization:
                visualization_cam_infos.append({
                    'keys': cam_dict['keys'],
                    'initial_cam': initial_cams.detach().float().cpu(),
                    'refined_cam': cam_dict['refined_cam'].detach().float().cpu(),
                })
            
            cams = cam_dict['refined_cam'].cuda()

            bg_score = torch.pow(1 - torch.max(cams, dim=0, keepdims=True)[0], self.cam_bg_thres).cuda()
            cams = torch.cat([bg_score, cams], dim=0).cuda()
            
            valid_key = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')
            valid_key = torch.from_numpy(valid_key).cuda()
            
            with torch.no_grad():
                cam_labels = _refine_cams(self.par, img[i], cams, valid_key)
            
            cam_list.append(cam_labels)

        all_cam_labels = torch.stack(cam_list, dim=0)

        visualization_data = None
        if return_visualization:
            visualization_data = {
                'cam_infos': visualization_cam_infos,
                'seg_clip': seg_clip.detach().float().cpu(),
                'seg_dino': (
                    seg_dino.detach().float().cpu()
                    if seg_dino is not None else None
                ),
                'seg_fused': seg.detach().float().cpu(),
            }

        if mode == 'train' and return_refined_cams:
            outputs = (
                seg, seg_clip, seg_dino, all_cam_labels, attn_pred,
                refined_cam_infos,
            )
            return outputs + ((visualization_data,) if return_visualization else ())
        if mode == 'train':
            return seg, seg_clip, seg_dino, all_cam_labels, attn_pred
        if return_refined_cams:
            outputs = (seg, all_cam_labels, attn_pred, refined_cam_infos)
            return outputs + ((visualization_data,) if return_visualization else ())

        return seg, all_cam_labels, attn_pred

        
