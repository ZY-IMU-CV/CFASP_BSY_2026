import argparse
import datetime
import logging
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '3')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import random
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WECLIP_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, WECLIP_ROOT)
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from utils.imutils import denormalize_img
from utils.visualization import save_voc_validation_visualization
from utils.voc_bpcr import (
    build_online_sgfr,
    generate_online_sgfr_labels,
    get_ablation_options,
    get_bpcr_internal_options,
    get_validation_visualization_options,
)
from clip.clip_text import class_names as VOC_CLASS_NAMES
from WeCLIP_model.model_attn_aff_voc import WeCLIP
from dice_loss import DiceLoss


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/home/stu_24/bsy/WeCLIP+SAM/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
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

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def get_segformer_adapter_enabled(cfg):
    segformer_cfg = cfg.get('segformer_head', None)
    if segformer_cfg is None:
        return False
    return bool(segformer_cfg.get('use_adapter', False))


def get_mutual_distill_options(cfg):
    distill_cfg = cfg.get('mutual_distill', None)
    if distill_cfg is None:
        return False, 0.1, 0
    return (
        bool(distill_cfg.get('enabled', False)),
        float(distill_cfg.get('weight', 0.1)),
        int(distill_cfg.get('start_iter', 0)),
    )


def get_validation_mode(cfg):
    val_cfg = cfg.get('val', None)
    mode = 'every' if val_cfg is None else str(val_cfg.get('mode', 'every')).lower()
    valid_modes = {'every', 'first_only', 'disabled'}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported val.mode={mode!r}. Expected one of {sorted(valid_modes)}."
        )
    return mode


def should_run_validation(current_iter, eval_interval, mode):
    if current_iter % eval_interval != 0:
        return False
    if mode == 'every':
        return True
    if mode == 'first_only':
        return current_iter == eval_interval
    return False


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


def validate(model=None, data_loader=None, cfg=None, online_sgfr=None,
             bpcr_enabled=False, use_box_prompt=True,
             use_point_prompt=True, use_joint_prompt=True,
             current_iter=None):

    model.eval()
    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    sgfr_preds, sgfr_gts = [], []
    sgfr_train_preds, sgfr_train_gts = [], []
    sgfr_eval_seen = 0
    sgfr_valid_count = 0
    sgfr_fg_ratios = []
    sgfr_eval_enabled = (
        bpcr_enabled
        and online_sgfr is not None
        and getattr(cfg.online_sgfr, 'eval_enabled', False)
    )
    sgfr_eval_samples = int(getattr(cfg.online_sgfr, 'eval_samples', 0))
    visualization_options = get_validation_visualization_options(cfg)
    visualization_count = 0
    visualization_root = None
    if visualization_options['enabled']:
        iteration_dir = (
            f"iter_{int(current_iter):06d}"
            if current_iter is not None else 'validation'
        )
        configured_dir = visualization_options['output_dir']
        visualization_root = (
            configured_dir
            if os.path.isabs(configured_dir)
            else os.path.join(cfg.work_dir.dir, configured_dir)
        )
        visualization_root = os.path.join(visualization_root, iteration_dir)
        os.makedirs(visualization_root, exist_ok=True)

    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()
        cls_label = cls_label.cuda()

        sample_name = str(name[0])
        selected_names = visualization_options['sample_names']
        within_visualization_limit = (
            visualization_options['max_samples'] <= 0
            or visualization_count < visualization_options['max_samples']
        )
        visualize_this_batch = (
            visualization_options['enabled']
            and within_visualization_limit
            and (not selected_names or sample_name in selected_names)
        )

        eval_sgfr_this_batch = (
            sgfr_eval_enabled
            and (sgfr_eval_samples <= 0 or sgfr_eval_seen < sgfr_eval_samples)
        )
        need_cam_details = eval_sgfr_this_batch or visualize_this_batch

        outputs = model(
            inputs,
            name,
            'val',
            # Keep image-level supervision identical in all ablation variants;
            # never infer present classes from a validation segmentation mask.
            cls_labels=cls_label,
            return_refined_cams=need_cam_details,
            return_visualization=visualize_this_batch,
        )

        model_visualization = None
        if need_cam_details:
            if visualize_this_batch:
                (
                    segs,
                    cam,
                    attn_loss,
                    refined_cam_infos,
                    model_visualization,
                ) = outputs
            else:
                segs, cam, attn_loss, refined_cam_infos = outputs
        else:
            segs, cam, attn_loss = outputs

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        resized_cam = resize_discrete_labels(cam, labels.shape[-2:])

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(resized_cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        run_bpcr_this_batch = (
            bpcr_enabled
            and online_sgfr is not None
            and (eval_sgfr_this_batch or visualize_this_batch)
        )
        prompt_visualization_infos = None
        if run_bpcr_this_batch:
            sgfr_outputs = generate_online_sgfr_labels(
                online_sgfr,
                inputs.detach().cpu(),
                refined_cam_infos,
                fallback_labels=cam,
                use_box_prompt=use_box_prompt,
                use_point_prompt=use_point_prompt,
                joint_box_point_prompt=use_joint_prompt,
                ignore_index=cfg.dataset.ignore_index,
                return_raw_labels=True,
                return_visualization=visualize_this_batch,
            )
            if visualize_this_batch:
                (
                    sgfr_train_label,
                    sgfr_count,
                    sgfr_fg_ratio,
                    sgfr_raw_label,
                    sgfr_valid_flags,
                    prompt_visualization_infos,
                ) = sgfr_outputs
            else:
                (
                    sgfr_train_label,
                    sgfr_count,
                    sgfr_fg_ratio,
                    sgfr_raw_label,
                    sgfr_valid_flags,
                ) = sgfr_outputs

        if eval_sgfr_this_batch:
            labels_np = labels.cpu().numpy().astype(np.int16)
            sgfr_train_np = resize_discrete_labels(
                sgfr_train_label, labels.shape[-2:]
            ).cpu().numpy().astype(np.int16)
            sgfr_raw_np = resize_discrete_labels(
                sgfr_raw_label, labels.shape[-2:]
            ).cpu().numpy().astype(np.int16)

            sgfr_train_preds += list(sgfr_train_np)
            sgfr_train_gts += list(labels_np)
            for sample_idx, is_valid in enumerate(sgfr_valid_flags):
                if is_valid:
                    sgfr_preds.append(sgfr_raw_np[sample_idx])
                    sgfr_gts.append(labels_np[sample_idx])

            sgfr_eval_seen += inputs.size(0)
            sgfr_valid_count += sgfr_count
            if sgfr_count > 0:
                sgfr_fg_ratios.append(sgfr_fg_ratio)

        if visualize_this_batch:
            if model_visualization is None:
                raise RuntimeError(
                    "Validation visualization was requested, but the model "
                    "did not return visualization tensors."
                )

            if prompt_visualization_infos is None:
                fallback = cam[0].detach().cpu().numpy().astype(np.uint8)
                prompt_visualization_infos = [{
                    'boxes': {},
                    'points': {},
                    'sam_pseudo_label': np.zeros_like(fallback),
                    'training_pseudo_label': fallback,
                    'used_cam_fallback': True,
                }]

            rgb_image = (
                denormalize_img(inputs.detach().cpu())[0]
                .permute(1, 2, 0)
                .numpy()
            )
            seg_dino_visualization = model_visualization['seg_dino']
            panel_path = save_voc_validation_visualization(
                output_root=visualization_root,
                sample_name=sample_name,
                image=rgb_image,
                cam_info=model_visualization['cam_infos'][0],
                prompt_info=prompt_visualization_infos[0],
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
            logging.info(
                "Saved validation visualization %d: %s",
                visualization_count,
                panel_path,
            )

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    sgfr_score = None
    sgfr_train_score = None
    if sgfr_preds:
        sgfr_score = evaluate.pseudo_scores(sgfr_gts, sgfr_preds, num_classes=cfg.dataset.num_classes)
    if sgfr_train_preds:
        sgfr_train_score = evaluate.pseudo_scores(sgfr_train_gts, sgfr_train_preds, num_classes=cfg.dataset.num_classes)
    sgfr_stats = {
        'val_split': cfg.val.split,
        'eval_seen': sgfr_eval_seen,
        'valid': sgfr_valid_count,
        'avg_fg_ratio': float(np.mean(sgfr_fg_ratios)) if sgfr_fg_ratios else 0.0,
    }
    model.train()
    return seg_score, cam_score, sgfr_score, sgfr_train_score, sgfr_stats


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask


def train(cfg):

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    
    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.val.split,
        stage='train',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False,
                            drop_last=False)

    deca_enabled, bpcr_requested = get_ablation_options(cfg)
    (
        use_box_prompt,
        use_point_prompt,
        use_joint_prompt,
    ) = get_bpcr_internal_options(cfg, bpcr_requested)
    # Box/Point both disabled is the no-BPCR row: use refined CAM directly and
    # do not initialize SAM or Grounding-DINO.
    bpcr_enabled = bpcr_requested and (use_box_prompt or use_point_prompt)
    adapter_requested = get_segformer_adapter_enabled(cfg)
    use_segformer_adapter = deca_enabled and adapter_requested
    WeCLIP_model = WeCLIP(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        device='cuda',
        use_segformer_adapter=use_segformer_adapter,
        use_deca=deca_enabled
    )
    logging.info(
        "VOC ablation: DECA=%s, BPCR=%s",
        "enabled" if deca_enabled else "disabled (original single-CLIP decoder)",
        "enabled" if bpcr_enabled else "disabled (refined-CAM fallback)",
    )
    logging.info(
        "BPCR internal ablation: Box=%s, Point=%s, Joint=%s",
        use_box_prompt,
        use_point_prompt,
        use_joint_prompt,
    )
    logging.info("SegFormerHead adapter MLP: %s", "enabled" if use_segformer_adapter else "disabled")
    logging.info("WECLIP+ DINO decoder branch: %s", "enabled" if deca_enabled else "disabled")
    logging.info('\nNetwork config: \n%s'%(WeCLIP_model))
    param_groups = WeCLIP_model.get_param_groups()
    WeCLIP_model.cuda()
    online_sgfr = build_online_sgfr(
        cfg,
        bpcr_enabled=bpcr_enabled,
        use_box_prompt=use_box_prompt,
    )

    # SAM3 is allowed to use BF16 only inside its explicitly scoped inference
    # calls. Fail fast if a global autocast/inference state leaks into WeCLIP
    # training, because that silently changes optimization and CAM quality.
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
            "WeCLIP training must run outside SAM3's BF16 autocast context."
        )
    if not grad_enabled:
        raise RuntimeError(
            "Gradient tracking is disabled after SGFR initialization. "
            "SAM3 inference_mode must be scoped to SAM3 calls only."
        )


    mask_size = int(cfg.dataset.crop_size // 16)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer_params = [
        {
            "params": param_groups[0],
            "lr": cfg.optimizer.learning_rate,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": param_groups[1],
            "lr": 0.0,
            "weight_decay": 0.0,
        },
        {
            "params": param_groups[2],
            "lr": cfg.optimizer.learning_rate*10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
        {
            "params": param_groups[3],
            "lr": cfg.optimizer.learning_rate*10,
            "weight_decay": cfg.optimizer.weight_decay,
        },
    ]
    optimizer = PolyWarmupAdamW(
        params=optimizer_params,
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()
    mutual_distill_enabled, mutual_distill_weight, mutual_distill_start_iter = get_mutual_distill_options(cfg)
    if mutual_distill_enabled and not deca_enabled:
        logging.info("Mutual distillation disabled because DECA is disabled.")
        mutual_distill_enabled = False
    validation_mode = get_validation_mode(cfg)
    logging.info(
        "Validation schedule: mode=%s, interval=%d",
        validation_mode,
        cfg.train.eval_iters
    )


    for n_iter in range(cfg.train.max_iters):
        
        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        use_online_sgfr_iter = (
            bpcr_enabled
            and (n_iter + 1) >= cfg.online_sgfr.start_iter
            and ((n_iter + 1 - cfg.online_sgfr.start_iter) % cfg.online_sgfr.interval == 0)
        )

        outputs = WeCLIP_model(
            inputs.cuda(),
            img_name,
            cls_labels=cls_labels,
            return_refined_cams=use_online_sgfr_iter
        )

        if use_online_sgfr_iter:
            segs, segs_clip, segs_dino, cam, attn_pred, refined_cam_infos = outputs
        else:
            segs, segs_clip, segs_dino, cam, attn_pred = outputs

        pseudo_label = cam
        online_sgfr_count = 0
        online_sgfr_fg_ratio = 0.0
        if use_online_sgfr_iter:
            pseudo_label, online_sgfr_count, online_sgfr_fg_ratio = generate_online_sgfr_labels(
                online_sgfr,
                inputs,
                refined_cam_infos,
                fallback_labels=cam,
                use_box_prompt=use_box_prompt,
                use_point_prompt=use_point_prompt,
                joint_box_point_prompt=use_joint_prompt,
                ignore_index=cfg.dataset.ignore_index
            )

        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        aff_source_label = pseudo_label.clone()

            
        aff_label = cams_to_affinity_label(aff_source_label, mask=attn_mask, ignore_index=cfg.dataset.ignore_index)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)

        dice_loss = criterion_dice(segs, pseudo_label.type(torch.long))

        loss = 1 * seg_loss + 1 * dice_loss + 0.1 * attn_loss
        if mutual_distill_enabled and (n_iter + 1) >= mutual_distill_start_iter:
            segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
            segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

            with torch.no_grad():
                pred_label_clip = torch.argmax(F.softmax(segs_clip, dim=1), dim=1)
                pred_label_dino = torch.argmax(F.softmax(segs_dino, dim=1), dim=1)

            seg_clip_loss2 = get_seg_loss(
                segs_clip,
                pred_label_dino.type(torch.long),
                ignore_index=cfg.dataset.ignore_index
            )
            seg_dino_loss2 = get_seg_loss(
                segs_dino,
                pred_label_clip.type(torch.long),
                ignore_index=cfg.dataset.ignore_index
            )
            loss = loss + mutual_distill_weight * (seg_clip_loss2 + seg_dino_loss2)


        avg_meter.add({
            'seg_loss': seg_loss.item(),
            'dice_loss': dice_loss.item(),
            'attn_loss': attn_loss.item(),
        })

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (n_iter + 1) % cfg.train.log_iters == 0:
            
            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size


            if bpcr_enabled:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, dice_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f, online_sgfr: %d/%d, sgfr_fg: %.3f"%(
                    n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('dice_loss'),
                    avg_meter.pop('attn_loss'), seg_mAcc,
                    online_sgfr_count, inputs.size(0), online_sgfr_fg_ratio))
            else:
                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, dice_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(
                    n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('dice_loss'),
                    avg_meter.pop('attn_loss'), seg_mAcc))

            writer.add_scalars(
                'train/loss',
                {
                    "seg_loss": seg_loss.item(),
                    "dice_loss": dice_loss.item(),
                    "attn_loss": attn_loss.item(),
                },
                global_step=n_iter,
            )

        
        current_iter = n_iter + 1
        if current_iter % cfg.train.eval_iters == 0:
            ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "WeCLIP_model_iter_%d.pth"%(n_iter+1))
            if current_iter > 26000:
                torch.save(WeCLIP_model.state_dict(), ckpt_name)

        if should_run_validation(current_iter, cfg.train.eval_iters, validation_mode):
            logging.info('Validating...')
            seg_score, cam_score, sgfr_score, sgfr_train_score, sgfr_stats = validate(
                model=WeCLIP_model,
                data_loader=val_loader,
                cfg=cfg,
                online_sgfr=online_sgfr,
                bpcr_enabled=bpcr_enabled,
                use_box_prompt=use_box_prompt,
                use_point_prompt=use_point_prompt,
                use_joint_prompt=use_joint_prompt,
                current_iter=current_iter,
            )
            logging.info("cams score:")
            logging.info(cam_score)
            logging.info("segs score:")
            logging.info(seg_score)
            if bpcr_enabled and getattr(cfg.online_sgfr, 'eval_enabled', False):
                logging.info("SGFR score:")
                logging.info(sgfr_score if sgfr_score is not None else "No valid SGFR pseudo labels were accepted.")
                logging.info("sgfr_train_label score:")
                logging.info(sgfr_train_score if sgfr_train_score is not None else "No SGFR train labels were evaluated.")
                logging.info("sgfr eval stats:")
                logging.info(sgfr_stats)

    return True


if __name__ == "__main__":

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.crop_size = args.crop_size

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    train(cfg=cfg)
