import argparse
import datetime
import logging
import os
import random
import sys
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WECLIP_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, WECLIP_ROOT)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clip.clip_text import class_names_coco as COCO_CLASS_NAMES
from datasets import coco
from dice_loss import DiceLoss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.imutils import denormalize_img
from utils.losses import get_aff_loss
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_model.model_attn_aff_coco import WeCLIP


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default=os.path.join(WECLIP_ROOT, "configs", "coco_attn_reg.yaml"),
    type=str,
    help="config",
)
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")
parser.add_argument("--crop_size", default=320, type=int, help="crop_size")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logger(filename="test.log"):
    log_formatter = logging.Formatter(
        "%(asctime)s - %(filename)s - %(levelname)s: %(message)s"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename, mode="w")
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = time_now - time0
    eta = delta * scale
    return str(delta), str((time_now + eta).replace(microsecond=0) - time_now)


def get_segformer_adapter_enabled(cfg):
    segformer_cfg = cfg.get("segformer_head", None)
    return False if segformer_cfg is None else bool(
        segformer_cfg.get("use_adapter", False)
    )


def get_mutual_distill_options(cfg):
    distill_cfg = cfg.get("mutual_distill", None)
    if distill_cfg is None:
        return False, 0.1, 0
    return (
        bool(distill_cfg.get("enabled", False)),
        float(distill_cfg.get("weight", 0.1)),
        int(distill_cfg.get("start_iter", 0)),
    )


def get_validation_mode(cfg):
    val_cfg = cfg.get("val", None)
    mode = "every" if val_cfg is None else str(
        val_cfg.get("mode", "every")
    ).lower()
    valid_modes = {"every", "first_only", "disabled"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported val.mode={mode!r}. Expected one of {sorted(valid_modes)}."
        )
    return mode


def should_run_validation(current_iter, eval_interval, mode):
    if current_iter % eval_interval != 0:
        return False
    if mode == "every":
        return True
    if mode == "first_only":
        return current_iter == eval_interval
    return False


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label != 0] = ignore_index
    bg_loss = F.cross_entropy(
        pred, bg_label.long(), ignore_index=ignore_index
    )
    fg_label = label.clone()
    fg_label[label == 0] = ignore_index
    fg_loss = F.cross_entropy(
        pred, fg_label.long(), ignore_index=ignore_index
    )
    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask = np.zeros((hw, hw))
    for i in range(hw):
        center_h = i // w
        center_w = i % w
        h0 = max(0, center_h - radius)
        h1 = min(h, center_h + radius + 1)
        w0 = max(0, center_w - radius)
        w1 = min(w, center_w + radius + 1)
        for neighbor_h in range(h0, h1):
            for neighbor_w in range(w0, w1):
                neighbor = neighbor_h * w + neighbor_w
                mask[i, neighbor] = 1
                mask[neighbor, i] = 1
    return mask


def update_pseudo_hist(hist, label_trues, label_preds, num_classes):
    """Accumulate pseudo-label metrics without retaining full COCO masks."""
    for label_true, label_pred in zip(label_trues, label_preds):
        label_true = label_true.reshape(-1).copy()
        label_pred = label_pred.reshape(-1).copy()
        ignored = label_pred == 255
        label_true[ignored] = 255
        label_pred[ignored] = 0
        valid = (
            (label_true >= 0)
            & (label_true < num_classes)
            & (label_pred >= 0)
            & (label_pred < num_classes)
        )
        encoded = (
            num_classes * label_true[valid].astype(np.int64)
            + label_pred[valid].astype(np.int64)
        )
        hist += np.bincount(
            encoded, minlength=num_classes**2
        ).reshape(num_classes, num_classes)
    return hist


def build_online_sgfr(cfg):
    sgfr_cfg = cfg.online_sgfr
    if not sgfr_cfg.enabled:
        return None

    sam_backend = str(getattr(sgfr_cfg, "sam_backend", "sam3")).lower()
    if sam_backend not in {"sam1", "sam3"}:
        raise ValueError(
            f"Unsupported online_sgfr.sam_backend={sam_backend!r}; "
            "expected 'sam1' or 'sam3'."
        )
    if sam_backend == "sam1" and getattr(
        sgfr_cfg, "sam_checkpoint", None
    ) is None:
        raise ValueError(
            "online_sgfr.sam_backend='sam1' requires online_sgfr.sam_checkpoint"
        )
    sam3_checkpoint = getattr(sgfr_cfg, "sam3_checkpoint", None)
    if sam_backend == "sam3" and sam3_checkpoint is None:
        raise ValueError(
            "online_sgfr.sam_backend='sam3' requires a local sam3_checkpoint"
        )
    if sgfr_cfg.gdino_checkpoint is None or sgfr_cfg.gdino_config is None:
        raise ValueError(
            "online_sgfr.enabled requires gdino_config and gdino_checkpoint"
        )

    from modules.sgfr import create_sgfr

    sgfr = create_sgfr(
        sam_backend=sam_backend,
        sam_checkpoint=getattr(sgfr_cfg, "sam_checkpoint", None),
        sam_model_type=getattr(sgfr_cfg, "sam_model_type", "vit_h"),
        sam3_checkpoint=sam3_checkpoint,
        sam3_bpe_path=getattr(sgfr_cfg, "sam3_bpe_path", None),
        sam3_resolution=int(getattr(sgfr_cfg, "sam3_resolution", 1008)),
        sam3_multimask_output=bool(
            getattr(sgfr_cfg, "sam3_multimask_output", False)
        ),
        sam3_compile=bool(getattr(sgfr_cfg, "sam3_compile", False)),
        gdino_checkpoint=sgfr_cfg.gdino_checkpoint,
        gdino_config=sgfr_cfg.gdino_config,
        gdino_text_encoder_type=getattr(
            sgfr_cfg, "gdino_text_encoder_type", None
        ),
        device="cuda",
        use_sam=True,
        use_gdino=True,
        box_threshold=sgfr_cfg.box_threshold,
        text_threshold=sgfr_cfg.text_threshold,
        cam_peak_threshold=sgfr_cfg.cam_peak_threshold,
    )
    if not sgfr.use_sam or not sgfr.sam_is_ready:
        raise RuntimeError(
            f"online_sgfr.enabled=True but {sam_backend} is not active. "
            "Check the selected backend installation and checkpoint."
        )
    if not sgfr.use_gdino or sgfr.gdino_model is None:
        raise RuntimeError(
            "online_sgfr.enabled=True but Grounding-DINO is not active. "
            "Check installation, config, and checkpoint."
        )

    logging.info(
        "Online SGFR initialized with active %s box+point PVS and Grounding-DINO",
        sam_backend.upper(),
    )
    return sgfr


def refined_cam_info_to_sgfr_inputs(refined_cam_info):
    keys = np.asarray(refined_cam_info["keys"]).astype(np.int64)
    refined_cam = refined_cam_info["refined_cam"]
    if torch.is_tensor(refined_cam):
        refined_cam = refined_cam.detach().float().cpu().numpy()

    class_names = []
    class_ids = []
    cam_dict = {}
    for cam_idx, key in enumerate(keys):
        if key < 0 or key >= len(COCO_CLASS_NAMES):
            continue
        class_id = int(key) + 1
        class_names.append(COCO_CLASS_NAMES[int(key)])
        class_ids.append(class_id)
        cam_dict[class_id] = refined_cam[cam_idx]
    return class_names, class_ids, cam_dict


def generate_online_sgfr_labels(
    sgfr,
    inputs,
    refined_cam_infos,
    fallback_labels,
    use_cam_points=False,
    ignore_index=255,
    return_raw_labels=False,
    joint_box_point_prompt=False,
):
    rgb_images = (
        denormalize_img(inputs.detach().cpu()).permute(0, 2, 3, 1).numpy()
    )
    fallback_labels_cpu = fallback_labels.detach().cpu()
    pseudo_labels = []
    raw_labels = []
    valid_flags = []
    sgfr_count = 0
    accepted_fg_ratios = []

    for i, image in enumerate(rgb_images):
        fallback = fallback_labels_cpu[i].numpy().astype(np.uint8)
        pseudo_label = fallback
        raw_label = np.zeros_like(fallback, dtype=np.uint8)
        is_valid = False
        try:
            class_names, class_ids, cam_dict = refined_cam_info_to_sgfr_inputs(
                refined_cam_infos[i]
            )
            if class_ids:
                generated = sgfr.generate(
                    image=image,
                    class_names=class_names,
                    class_ids=class_ids,
                    cam=cam_dict if use_cam_points else None,
                    joint_box_point_prompt=joint_box_point_prompt,
                    ignore_index=ignore_index,
                )
                if generated.shape != fallback.shape:
                    generated = cv2.resize(
                        generated.astype(np.uint8),
                        (fallback.shape[1], fallback.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                class_ids_np = np.asarray(class_ids, dtype=np.uint8)
                foreground_mask = np.isin(generated, class_ids_np)
                fg_ratio = float(foreground_mask.mean())
                if foreground_mask.any():
                    generated = generated.astype(np.uint8)
                    raw_label = generated
                    pseudo_label = generated
                    is_valid = True
                    sgfr_count += 1
                    accepted_fg_ratios.append(fg_ratio)
        except Exception as error:
            warnings.warn(
                f"Online SGFR failed for sample {i}: {error}. "
                "Falling back to CAM label."
            )

        pseudo_labels.append(torch.from_numpy(pseudo_label.astype(np.int64)))
        raw_labels.append(torch.from_numpy(raw_label.astype(np.int64)))
        valid_flags.append(is_valid)

    avg_fg_ratio = (
        float(np.mean(accepted_fg_ratios)) if accepted_fg_ratios else 0.0
    )
    pseudo_labels = torch.stack(pseudo_labels, dim=0).to(
        fallback_labels.device
    )
    if return_raw_labels:
        raw_labels = torch.stack(raw_labels, dim=0).to(fallback_labels.device)
        return (
            pseudo_labels,
            sgfr_count,
            avg_fg_ratio,
            raw_labels,
            valid_flags,
        )
    return pseudo_labels, sgfr_count, avg_fg_ratio


def validate(model=None, data_loader=None, cfg=None, online_sgfr=None):
    num_classes = int(cfg.dataset.num_classes)
    preds, gts, cams = [], [], []
    seg_hist = np.zeros((num_classes, num_classes))
    cam_hist = np.zeros((num_classes, num_classes))
    sgfr_hist = np.zeros((num_classes, num_classes))
    sgfr_train_hist = np.zeros((num_classes, num_classes))
    sgfr_scored_count = 0
    sgfr_train_scored_count = 0
    sgfr_eval_seen = 0
    sgfr_valid_count = 0
    sgfr_fg_ratios = []
    sgfr_eval_enabled = (
        online_sgfr is not None
        and cfg.online_sgfr.enabled
        and bool(getattr(cfg.online_sgfr, "eval_enabled", False))
    )
    sgfr_eval_samples = int(getattr(cfg.online_sgfr, "eval_samples", 0))
    metric_chunk_size = max(1, int(cfg.val.get("metric_chunk_size", 50)))

    model.eval()
    for _, data in tqdm(
        enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="
    ):
        name, inputs, labels, cls_label = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_label = cls_label.cuda()

        eval_sgfr_this_batch = sgfr_eval_enabled and (
            sgfr_eval_samples <= 0 or sgfr_eval_seen < sgfr_eval_samples
        )
        outputs = model(
            inputs,
            name,
            "val",
            cls_labels=cls_label,
            return_refined_cams=eval_sgfr_this_batch,
        )
        if eval_sgfr_this_batch:
            segs, cam, _, refined_cam_infos = outputs
        else:
            segs, cam, _ = outputs

        resized_segs = F.interpolate(
            segs,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        preds.extend(
            torch.argmax(resized_segs, dim=1)
            .cpu()
            .numpy()
            .astype(np.int16)
        )
        cams.extend(cam.cpu().numpy().astype(np.int16))
        gts.extend(labels.cpu().numpy().astype(np.int16))

        if eval_sgfr_this_batch:
            (
                sgfr_train_label,
                sgfr_count,
                sgfr_fg_ratio,
                sgfr_raw_label,
                sgfr_valid_flags,
            ) = generate_online_sgfr_labels(
                online_sgfr,
                inputs.detach().cpu(),
                refined_cam_infos,
                fallback_labels=cam,
                use_cam_points=cfg.online_sgfr.use_cam_points,
                joint_box_point_prompt=bool(
                    getattr(
                        cfg.online_sgfr, "joint_box_point_prompt", False
                    )
                ),
                ignore_index=cfg.dataset.ignore_index,
                return_raw_labels=True,
            )
            labels_np = labels.cpu().numpy().astype(np.int16)
            sgfr_train_np = sgfr_train_label.cpu().numpy().astype(np.int16)
            sgfr_raw_np = sgfr_raw_label.cpu().numpy().astype(np.int16)
            sgfr_train_hist = update_pseudo_hist(
                sgfr_train_hist,
                labels_np,
                sgfr_train_np,
                num_classes,
            )
            sgfr_train_scored_count += len(labels_np)
            for sample_idx, is_valid in enumerate(sgfr_valid_flags):
                if is_valid:
                    sgfr_hist = update_pseudo_hist(
                        sgfr_hist,
                        [labels_np[sample_idx]],
                        [sgfr_raw_np[sample_idx]],
                        num_classes,
                    )
                    sgfr_scored_count += 1

            sgfr_eval_seen += inputs.size(0)
            sgfr_valid_count += sgfr_count
            if sgfr_count > 0:
                sgfr_fg_ratios.append(sgfr_fg_ratio)

        if len(gts) >= metric_chunk_size:
            seg_hist, _ = evaluate.scores(
                gts, preds, seg_hist, num_classes=num_classes
            )
            cam_hist, _ = evaluate.scores(
                gts, cams, cam_hist, num_classes=num_classes
            )
            preds, gts, cams = [], [], []

    seg_hist, seg_score = evaluate.scores(
        gts, preds, seg_hist, num_classes=num_classes
    )
    cam_hist, cam_score = evaluate.scores(
        gts, cams, cam_hist, num_classes=num_classes
    )
    sgfr_score = None
    sgfr_train_score = None
    if sgfr_scored_count:
        _, sgfr_score = evaluate.scores(
            [], [], sgfr_hist, num_classes=num_classes
        )
    if sgfr_train_scored_count:
        _, sgfr_train_score = evaluate.scores(
            [], [], sgfr_train_hist, num_classes=num_classes
        )
    sgfr_stats = {
        "val_split": cfg.val.split,
        "eval_seen": sgfr_eval_seen,
        "valid": sgfr_valid_count,
        "avg_fg_ratio": (
            float(np.mean(sgfr_fg_ratios)) if sgfr_fg_ratios else 0.0
        ),
    }
    model.train()
    return seg_score, cam_score, sgfr_score, sgfr_train_score, sgfr_stats


def train(cfg):
    num_workers = int(cfg.train.get("num_workers", 10))
    time0 = datetime.datetime.now().replace(microsecond=0)

    train_dataset = coco.CocoClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage="train",
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage="val",
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.samples_per_gpu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )

    use_segformer_adapter = get_segformer_adapter_enabled(cfg)
    weclip_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device="cuda",
        use_segformer_adapter=use_segformer_adapter,
    )
    logging.info(
        "SegFormerHead adapter MLP: %s",
        "enabled" if use_segformer_adapter else "disabled",
    )
    logging.info("WeCLIP + DINO decoder branch: enabled")
    logging.info("\nNetwork config: \n%s", weclip_model)
    param_groups = weclip_model.get_param_groups()
    weclip_model.cuda()
    online_sgfr = build_online_sgfr(cfg)

    try:
        cuda_autocast_enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        cuda_autocast_enabled = torch.is_autocast_enabled()
    grad_enabled = torch.is_grad_enabled()
    logging.info(
        "Precision state after SGFR init: cuda_autocast=%s, grad_enabled=%s",
        cuda_autocast_enabled,
        grad_enabled,
    )
    if cuda_autocast_enabled:
        raise RuntimeError(
            "CUDA autocast leaked globally after SGFR initialization. "
            "WeCLIP training must remain FP32 outside SAM3 calls."
        )
    if not grad_enabled:
        raise RuntimeError(
            "Gradient tracking is disabled after SGFR initialization."
        )

    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(
        h=mask_size, w=mask_size, radius=args.radius
    )
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)
    optimizer_params = [
        {
            "params": param_groups[0],
            "lr": cfg.optimizer.learning_rate,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {"params": param_groups[1], "lr": 0.0, "weight_decay": 0.0},
        {
            "params": param_groups[2],
            "lr": cfg.optimizer.learning_rate * 10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": param_groups[3],
            "lr": cfg.optimizer.learning_rate * 10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
    ]
    optimizer = PolyWarmupAdamW(
        params=optimizer_params,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power,
    )
    logging.info("\nOptimizer: \n%s", optimizer)

    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()
    criterion_dice = DiceLoss().cuda()
    (
        mutual_distill_enabled,
        mutual_distill_weight,
        mutual_distill_start_iter,
    ) = get_mutual_distill_options(cfg)
    validation_mode = get_validation_mode(cfg)
    save_start_iter = int(cfg.train.get("save_start_iter", 40000))
    best_seg_miou = -1.0
    logging.info(
        "Validation schedule: mode=%s, interval=%d",
        validation_mode,
        cfg.train.eval_iters,
    )

    for n_iter in range(cfg.train.max_iters):
        try:
            img_name, inputs, cls_labels, _ = next(train_loader_iter)
        except StopIteration:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, _ = next(train_loader_iter)

        current_iter = n_iter + 1
        use_online_sgfr_iter = (
            cfg.online_sgfr.enabled
            and current_iter >= cfg.online_sgfr.start_iter
            and (
                (current_iter - cfg.online_sgfr.start_iter)
                % cfg.online_sgfr.interval
                == 0
            )
        )
        outputs = weclip_model(
            inputs.cuda(),
            img_name,
            cls_labels=cls_labels,
            return_refined_cams=use_online_sgfr_iter,
        )
        if use_online_sgfr_iter:
            (
                segs,
                segs_clip,
                segs_dino,
                cam,
                attn_pred,
                refined_cam_infos,
            ) = outputs
        else:
            segs, segs_clip, segs_dino, cam, attn_pred = outputs

        pseudo_label = cam
        online_sgfr_count = 0
        online_sgfr_fg_ratio = 0.0
        if use_online_sgfr_iter:
            (
                pseudo_label,
                online_sgfr_count,
                online_sgfr_fg_ratio,
            ) = generate_online_sgfr_labels(
                online_sgfr,
                inputs,
                refined_cam_infos,
                fallback_labels=cam,
                use_cam_points=cfg.online_sgfr.use_cam_points,
                joint_box_point_prompt=bool(
                    getattr(
                        cfg.online_sgfr, "joint_box_point_prompt", False
                    )
                ),
                ignore_index=cfg.dataset.ignore_index,
            )

        segs = F.interpolate(
            segs,
            size=pseudo_label.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        aff_label = cams_to_affinity_label(
            pseudo_label.clone(),
            mask=attn_mask,
            ignore_index=cfg.dataset.ignore_index,
        )
        attn_loss, _, _ = get_aff_loss(attn_pred, aff_label)
        seg_loss = get_seg_loss(
            segs, pseudo_label.long(), ignore_index=cfg.dataset.ignore_index
        )
        dice_loss = criterion_dice(segs, pseudo_label.clone().long())
        loss = seg_loss + dice_loss + 0.1 * attn_loss

        if (
            mutual_distill_enabled
            and current_iter >= mutual_distill_start_iter
        ):
            segs_clip = F.interpolate(
                segs_clip,
                size=pseudo_label.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            segs_dino = F.interpolate(
                segs_dino,
                size=pseudo_label.shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            with torch.no_grad():
                pred_label_clip = torch.argmax(segs_clip, dim=1)
                pred_label_dino = torch.argmax(segs_dino, dim=1)
            seg_clip_loss2 = get_seg_loss(
                segs_clip,
                pred_label_dino.long(),
                ignore_index=cfg.dataset.ignore_index,
            )
            seg_dino_loss2 = get_seg_loss(
                segs_dino,
                pred_label_clip.long(),
                ignore_index=cfg.dataset.ignore_index,
            )
            loss = loss + mutual_distill_weight * (
                seg_clip_loss2 + seg_dino_loss2
            )

        avg_meter.add(
            {
                "seg_loss": seg_loss.item(),
                "dice_loss": dice_loss.item(),
                "attn_loss": attn_loss.item(),
            }
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if current_iter % cfg.train.log_iters == 0:
            delta, eta = cal_eta(time0, current_iter, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]["lr"]
            preds = torch.argmax(segs, dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)
            valid = gts != cfg.dataset.ignore_index
            seg_macc = (
                float((preds[valid] == gts[valid]).mean())
                if valid.any()
                else 0.0
            )
            log_values = (
                current_iter,
                delta,
                eta,
                cur_lr,
                avg_meter.pop("seg_loss"),
                avg_meter.pop("dice_loss"),
                avg_meter.pop("attn_loss"),
                seg_macc,
            )
            if cfg.online_sgfr.enabled:
                logging.info(
                    "Iter: %d; Elapsed: %s; ETA: %s; LR: %.3e; "
                    "pseudo_seg_loss: %.4f, dice_loss: %.4f, "
                    "attn_loss: %.4f, pseudo_seg_mAcc: %.4f, "
                    "online_sgfr: %d/%d, sgfr_fg: %.3f",
                    *log_values,
                    online_sgfr_count,
                    inputs.size(0),
                    online_sgfr_fg_ratio,
                )
            else:
                logging.info(
                    "Iter: %d; Elapsed: %s; ETA: %s; LR: %.3e; "
                    "pseudo_seg_loss: %.4f, dice_loss: %.4f, "
                    "attn_loss: %.4f, pseudo_seg_mAcc: %.4f",
                    *log_values,
                )
            writer.add_scalars(
                "train/loss",
                {
                    "seg_loss": seg_loss.item(),
                    "dice_loss": dice_loss.item(),
                    "attn_loss": attn_loss.item(),
                },
                global_step=current_iter,
            )

        if (
            current_iter % cfg.train.eval_iters == 0
            and current_iter >= save_start_iter
        ):
            ckpt_name = os.path.join(
                cfg.work_dir.ckpt_dir,
                f"WeCLIP_model_iter_{current_iter}.pth",
            )
            torch.save(weclip_model.state_dict(), ckpt_name)

        if should_run_validation(
            current_iter, cfg.train.eval_iters, validation_mode
        ):
            logging.info("Validating COCO...")
            (
                seg_score,
                cam_score,
                sgfr_score,
                sgfr_train_score,
                sgfr_stats,
            ) = validate(
                model=weclip_model,
                data_loader=val_loader,
                cfg=cfg,
                online_sgfr=online_sgfr,
            )
            logging.info("COCO CAM score: %s", cam_score)
            logging.info("COCO segmentation score: %s", seg_score)
            writer.add_scalars(
                "val/miou",
                {"seg": seg_score["miou"], "cam": cam_score["miou"]},
                global_step=current_iter,
            )
            if seg_score["miou"] > best_seg_miou:
                best_seg_miou = seg_score["miou"]
                torch.save(
                    weclip_model.state_dict(),
                    os.path.join(
                        cfg.work_dir.ckpt_dir, "WeCLIP_model_best.pth"
                    ),
                )
                logging.info("New best COCO mIoU: %.4f", best_seg_miou)

            if (
                cfg.online_sgfr.enabled
                and getattr(cfg.online_sgfr, "eval_enabled", False)
            ):
                logging.info(
                    "SGFR raw pseudo-label score: %s",
                    sgfr_score
                    if sgfr_score is not None
                    else "No valid SGFR labels were accepted.",
                )
                logging.info(
                    "SGFR train-label score: %s",
                    sgfr_train_score
                    if sgfr_train_score is not None
                    else "No SGFR train labels were evaluated.",
                )
                logging.info("SGFR validation stats: %s", sgfr_stats)

    writer.close()
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(
        cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp
    )
    cfg.work_dir.pred_dir = os.path.join(
        cfg.work_dir.dir, cfg.work_dir.pred_dir
    )
    cfg.work_dir.tb_logger_dir = os.path.join(
        cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp
    )
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp + ".log"))
    logging.info("\nargs: %s", args)
    logging.info("\nconfigs: %s", cfg)
    setup_seed(1)
    train(cfg=cfg)
