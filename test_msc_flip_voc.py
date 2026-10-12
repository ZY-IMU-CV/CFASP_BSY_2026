import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WECLIP_ROOT = SCRIPT_DIR
sys.path.insert(0, WECLIP_ROOT)
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap, denormalize_img
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from WeCLIP_model.model_attn_aff_voc import WeCLIP
from utils.voc_bpcr import (
    build_online_sgfr,
    generate_online_sgfr_labels,
    get_ablation_options,
    get_bpcr_internal_options,
    get_validation_visualization_options,
)
from utils.visualization import save_voc_validation_visualization
from clip.clip_text import class_names as VOC_CLASS_NAMES
import imageio.v2 as imageio
parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--resize_long", default=512, type=int, help="resize the long side")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set") #val
parser.add_argument("--model_path", default="/your/path/WeCLIP/WeCLIP_model_iter_30000.pth", type=str, help="model_path")
parser.add_argument("--scales", default="1.0,0.75", type=str, help="multi-scale factors, comma separated")
parser.add_argument("--crf", action="store_true", default=True, help="run DenseCRF after saving logits")
parser.add_argument("--no_crf", action="store_false", dest="crf", help="do not run DenseCRF after validation")
parser.add_argument("--only_crf", action="store_true", help="skip model inference and only run CRF from saved logits")
parser.add_argument("--crf_jobs", default=0, type=int, help="number of CRF worker jobs; 0 uses 80 percent of CPU cores")



def parse_scales(scales):
    return [float(scale) for scale in scales.split(",") if scale.strip()]


def get_segformer_adapter_enabled(cfg):
    segformer_cfg = cfg.get('segformer_head', None)
    if segformer_cfg is None:
        return False
    return bool(segformer_cfg.get('use_adapter', False))


def get_deca_enabled(cfg):
    ablation_cfg = cfg.get('ablation', None)
    return True if ablation_cfg is None else bool(ablation_cfg.get('deca', True))


def is_unlabeled_test_split(split):
    return str(split).strip().lower().startswith('test')


def resize_discrete_labels(label_map, target_size):
    """Resize BxHxW integer labels without creating interpolated class IDs."""
    if tuple(label_map.shape[-2:]) == tuple(target_size):
        return label_map
    original_dtype = label_map.dtype
    resized = F.interpolate(
        label_map.unsqueeze(1).float(),
        size=target_size,
        mode='nearest',
    )
    return resized.squeeze(1).to(dtype=original_dtype)


def validate(
    model,
    dataset,
    cfg,
    test_scales=None,
    online_sgfr=None,
    bpcr_enabled=False,
    use_box_prompt=True,
    use_point_prompt=True,
    use_joint_prompt=True,
):

    is_test_set = is_unlabeled_test_split(args.eval_set)
    _preds, _gts, _msc_preds, cams = [], [], [], []
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    # with torch.no_grad(), torch.cuda.device(0):
    model.cuda(0)
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    visualization_options = get_validation_visualization_options(cfg)
    if is_test_set and visualization_options['enabled']:
        print(
            "VOC test has no image-level class labels or segmentation GT; "
            "CAM/BPCR visualization is disabled for this split."
        )
        visualization_options['enabled'] = False
    visualization_count = 0
    visualization_root = None
    if visualization_options['enabled']:
        configured_dir = visualization_options['output_dir']
        visualization_root = (
            configured_dir
            if os.path.isabs(configured_dir)
            else os.path.join(args.work_dir, configured_dir)
        )
        visualization_root = os.path.join(visualization_root, 'inference')
        os.makedirs(visualization_root, exist_ok=True)

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_labels = None if is_test_set else cls_labels.cuda()

        sample_name = str(name[0])
        selected_names = visualization_options['sample_names']
        within_visualization_limit = (
            visualization_options['max_samples'] <= 0
            or visualization_count < visualization_options['max_samples']
        )
        visualize_this_sample = (
            visualization_options['enabled']
            and within_visualization_limit
            and (not selected_names or sample_name in selected_names)
        )

        #######
        # resize long side to 512
        
        _, _, h, w = inputs.shape
        ratio = args.resize_long / max(h,w)
        _h, _w = int(h*ratio), int(w*ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)
        
        #######

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        cls_labels_cat = (
            None
            if cls_labels is None
            else torch.cat([cls_labels, cls_labels], dim=0)
        )
        outputs = model(
            inputs_cat,
            names,
            mode='val',
            cls_labels=cls_labels_cat,
            return_refined_cams=visualize_this_sample,
            return_visualization=visualize_this_sample,
            skip_cam_generation=is_test_set,
        )
        if visualize_this_sample:
            (
                segs_cat,
                cam,
                attn_loss,
                refined_cam_infos,
                model_visualization,
            ) = outputs
        else:
            segs_cat, cam, attn_loss = outputs

        if visualize_this_sample:
            prompt_info = None
            if bpcr_enabled and online_sgfr is not None:
                sgfr_outputs = generate_online_sgfr_labels(
                    online_sgfr,
                    inputs[:1].detach().cpu(),
                    refined_cam_infos[:1],
                    fallback_labels=cam[:1],
                    use_box_prompt=use_box_prompt,
                    use_point_prompt=use_point_prompt,
                    joint_box_point_prompt=use_joint_prompt,
                    ignore_index=cfg.dataset.ignore_index,
                    return_raw_labels=True,
                    return_visualization=True,
                )
                prompt_info = sgfr_outputs[-1][0]

            if prompt_info is None:
                fallback = cam[0].detach().cpu().numpy().astype(np.uint8)
                prompt_info = {
                    'boxes': {},
                    'points': {},
                    'sam_pseudo_label': np.zeros_like(fallback),
                    'training_pseudo_label': fallback,
                    'used_cam_fallback': True,
                }

            rgb_image = (
                denormalize_img(inputs[:1].detach().cpu())[0]
                .permute(1, 2, 0)
                .numpy()
            )
            seg_dino_visualization = model_visualization['seg_dino']
            panel_path = save_voc_validation_visualization(
                output_root=visualization_root,
                sample_name=sample_name,
                image=rgb_image,
                cam_info=model_visualization['cam_infos'][0],
                prompt_info=prompt_info,
                seg_clip=model_visualization['seg_clip'][0],
                seg_dino=(
                    None
                    if seg_dino_visualization is None
                    else seg_dino_visualization[0]
                ),
                seg_fused=model_visualization['seg_fused'][0],
                class_names=VOC_CLASS_NAMES,
                overlay_alpha=visualization_options['overlay_alpha'],
            )
            visualization_count += 1
            print(
                f"Saved validation visualization {visualization_count}: "
                f"{panel_path}"
            )
        
        if cam is not None:
            cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, h, w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_cat, cam_cat, attn_loss = model(
                    inputs_cat,
                    names,
                    mode='val',
                    cls_labels=cls_labels_cat,
                    skip_cam_generation=is_test_set,
                )

                _segs_cat = F.interpolate(segs_cat, size=(h, w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)
        msc_pred_np = msc_seg_preds[0].cpu().numpy().astype(np.uint8)
        imageio.imsave(
            os.path.join(args.work_dir, 'prediction_msc', name[0] + '.png'),
            msc_pred_np,
        )
        imageio.imsave(
            os.path.join(
                args.work_dir, 'prediction_msc_cmap', name[0] + '.png'
            ),
            encode_cmap(msc_pred_np).astype(np.uint8),
        )

        if not is_test_set:
            resized_cam = resize_discrete_labels(cam, labels.shape[-2:])
            cams += list(resized_cam.cpu().numpy().astype(np.int16))
            _preds += list(seg_preds.cpu().numpy().astype(np.int16))
            _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
            _gts += list(labels.cpu().numpy().astype(np.int16))

        if not is_test_set and num % 100 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []


        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})
            
    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")
    is_test_set = is_unlabeled_test_split(args.eval_set)

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,   # 5
        bi_w=4,         # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        label = None
        if not is_test_set:
            label_name = os.path.join(labels_path, name + ".png")
            label = imageio.imread(label_name)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    n_jobs = args.crf_jobs if args.crf_jobs > 0 else int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    if is_test_set:
        print(
            f"VOC test inference complete: saved {len(results)} CRF predictions "
            f"to {os.path.join(args.work_dir, 'prediction')}"
        )
        return True

    preds, gts = zip(*results)
    hist = np.zeros((21, 21))
    hist, score = evaluate.scores(gts, preds, hist, 21)

    print("crf score:")
    print(score)
    
    return True

def main(cfg):
    is_test_set = is_unlabeled_test_split(args.eval_set)
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='test' if is_test_set else 'val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    if not args.only_crf:
        deca_enabled = get_deca_enabled(cfg)
        _, bpcr_requested = get_ablation_options(cfg)
        (
            use_box_prompt,
            use_point_prompt,
            use_joint_prompt,
        ) = get_bpcr_internal_options(cfg, bpcr_requested)
        visualization_enabled = (
            get_validation_visualization_options(cfg)['enabled']
            and not is_test_set
        )
        bpcr_enabled = (
            visualization_enabled
            and bpcr_requested
            and (use_box_prompt or use_point_prompt)
        )
        WeCLIP_model = WeCLIP(num_classes=cfg.dataset.num_classes,
                         clip_model=cfg.clip_init.clip_pretrain_path,
                         embedding_dim=cfg.clip_init.embedding_dim,
                         in_channels=cfg.clip_init.in_channels,
                         dataset_root_path=cfg.dataset.root_dir,
                         device='cuda',
                         use_segformer_adapter=(
                             deca_enabled and get_segformer_adapter_enabled(cfg)
                         ),
                         use_deca=deca_enabled)
        
        trained_state_dict = torch.load(args.model_path, map_location="cpu")

        WeCLIP_model.load_state_dict(state_dict=trained_state_dict, strict=False)
        WeCLIP_model.eval()

        online_sgfr = build_online_sgfr(
            cfg,
            bpcr_enabled=bpcr_enabled,
            use_box_prompt=use_box_prompt,
        )

        gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(
            model=WeCLIP_model,
            dataset=val_dataset,
            cfg=cfg,
            test_scales=parse_scales(args.scales),
            online_sgfr=online_sgfr,
            bpcr_enabled=bpcr_enabled,
            use_box_prompt=use_box_prompt,
            use_point_prompt=use_point_prompt,
            use_joint_prompt=use_joint_prompt,
        )
        torch.cuda.empty_cache()

        if is_test_set:
            print(
                "VOC test has no public ground truth; skipped CAM/segmentation "
                "metrics. Multi-scale predictions were saved to "
                f"{os.path.join(args.work_dir, 'prediction_msc')}."
            )
        else:
            preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
            msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
            cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

            print("cams score:")
            print(cam_score)
            print("segs score:")
            print(seg_score)
            print("msc segs score:")
            print(msc_seg_score)

    if args.crf or args.only_crf:
        crf_proc(config=cfg)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_msc", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_msc_cmap", exist_ok=True)

    main(cfg=cfg)
