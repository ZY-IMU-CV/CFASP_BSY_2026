import logging
import warnings

import cv2
import numpy as np
import torch

from clip.clip_text import class_names as VOC_CLASS_NAMES
from utils.imutils import denormalize_img


def get_ablation_options(cfg):
    """Return module-level VOC ablation switches with legacy defaults."""
    ablation_cfg = cfg.get('ablation', None)
    if ablation_cfg is None:
        online_sgfr_cfg = cfg.get('online_sgfr', None)
        legacy_bpcr_enabled = (
            online_sgfr_cfg is not None
            and bool(online_sgfr_cfg.get('enabled', False))
        )
        return True, legacy_bpcr_enabled
    return (
        bool(ablation_cfg.get('deca', True)),
        bool(ablation_cfg.get('bpcr', False)),
    )


def get_bpcr_internal_options(cfg, bpcr_enabled):
    """Resolve the Box/Point/Joint switches used by the BPCR ablation."""
    if not bpcr_enabled:
        return False, False, False

    ablation_cfg = cfg.get('ablation', None)
    internal_cfg = (
        None if ablation_cfg is None
        else ablation_cfg.get('bpcr_internal', None)
    )
    if internal_cfg is None:
        sgfr_cfg = cfg.get('online_sgfr', {})
        use_box_prompt = True
        use_point_prompt = bool(sgfr_cfg.get('use_cam_points', True))
        use_joint_prompt = bool(
            sgfr_cfg.get('joint_box_point_prompt', False)
        )
    else:
        use_box_prompt = bool(internal_cfg.get('box', True))
        use_point_prompt = bool(internal_cfg.get('point', True))
        use_joint_prompt = bool(internal_cfg.get('joint', True))

    if use_joint_prompt and not (use_box_prompt and use_point_prompt):
        raise ValueError(
            "ablation.bpcr_internal.joint=true requires both box=true "
            "and point=true"
        )
    return use_box_prompt, use_point_prompt, use_joint_prompt


def get_validation_visualization_options(cfg):
    val_cfg = cfg.get('val', {})
    vis_cfg = val_cfg.get('visualization', None)
    if vis_cfg is None:
        return {
            'enabled': False,
            'output_dir': 'visualizations',
            'max_samples': 0,
            'sample_names': set(),
            'overlay_alpha': 0.55,
        }
    return {
        'enabled': bool(vis_cfg.get('enabled', False)),
        'output_dir': str(vis_cfg.get('output_dir', 'visualizations')),
        'max_samples': int(vis_cfg.get('max_samples', 20)),
        'sample_names': {
            str(name) for name in vis_cfg.get('sample_names', [])
        },
        'overlay_alpha': float(vis_cfg.get('overlay_alpha', 0.55)),
    }


def build_online_sgfr(cfg, bpcr_enabled, use_box_prompt=True):
    if not bpcr_enabled:
        return None

    sgfr_cfg = cfg.online_sgfr
    sam_backend = str(getattr(sgfr_cfg, 'sam_backend', 'sam3')).lower()
    if sam_backend not in {'sam1', 'sam3'}:
        raise ValueError(
            f"Unsupported online_sgfr.sam_backend={sam_backend!r}; "
            "expected 'sam1' or 'sam3'."
        )
    if (
        sam_backend == 'sam1'
        and getattr(sgfr_cfg, 'sam_checkpoint', None) is None
    ):
        raise ValueError(
            "online_sgfr.sam_backend='sam1' requires "
            "online_sgfr.sam_checkpoint"
        )
    sam3_checkpoint = getattr(sgfr_cfg, 'sam3_checkpoint', None)
    if sam_backend == 'sam3' and sam3_checkpoint is None:
        raise ValueError(
            "online_sgfr.sam_backend='sam3' requires a local "
            "sam3_checkpoint because SAM 3 runs in offline mode"
        )

    gdino_checkpoint = (
        getattr(sgfr_cfg, 'gdino_checkpoint', None)
        if use_box_prompt else None
    )
    gdino_config = (
        getattr(sgfr_cfg, 'gdino_config', None)
        if use_box_prompt else None
    )
    if use_box_prompt and (gdino_checkpoint is None or gdino_config is None):
        raise ValueError(
            "BPCR box=true requires Grounding-DINO and therefore requires "
            "online_sgfr.gdino_config plus online_sgfr.gdino_checkpoint"
        )

    from modules.sgfr import create_sgfr

    sgfr = create_sgfr(
        sam_backend=sam_backend,
        sam_checkpoint=getattr(sgfr_cfg, 'sam_checkpoint', None),
        sam_model_type=getattr(sgfr_cfg, 'sam_model_type', 'vit_h'),
        sam3_checkpoint=sam3_checkpoint,
        sam3_bpe_path=getattr(sgfr_cfg, 'sam3_bpe_path', None),
        sam3_resolution=int(getattr(sgfr_cfg, 'sam3_resolution', 1008)),
        sam3_multimask_output=bool(
            getattr(sgfr_cfg, 'sam3_multimask_output', False)
        ),
        sam3_compile=bool(getattr(sgfr_cfg, 'sam3_compile', False)),
        gdino_checkpoint=gdino_checkpoint,
        gdino_config=gdino_config,
        gdino_text_encoder_type=(
            getattr(sgfr_cfg, 'gdino_text_encoder_type', None)
            if use_box_prompt else None
        ),
        device='cuda',
        use_sam=True,
        use_gdino=use_box_prompt,
        box_threshold=sgfr_cfg.box_threshold,
        text_threshold=sgfr_cfg.text_threshold,
        cam_peak_threshold=sgfr_cfg.cam_peak_threshold,
    )
    if not sgfr.use_sam or not sgfr.sam_is_ready:
        raise RuntimeError(
            f"ablation.bpcr=true but {sam_backend} is not active. "
            "Check the selected backend installation and checkpoint "
            "configuration."
        )
    if use_box_prompt and (
        not sgfr.use_gdino or sgfr.gdino_model is None
    ):
        raise RuntimeError(
            "BPCR box=true but Grounding-DINO is not active. Check the "
            "groundingdino installation, config, and checkpoint."
        )

    logging.info(
        "Online BPCR initialized: SAM=%s, Grounding-DINO=%s",
        sam_backend.upper(),
        "enabled" if use_box_prompt else "disabled (point-only)",
    )
    return sgfr


def refined_cam_info_to_sgfr_inputs(refined_cam_info):
    keys = np.asarray(refined_cam_info['keys']).astype(np.int64)
    refined_cam = refined_cam_info['refined_cam']
    if torch.is_tensor(refined_cam):
        refined_cam = refined_cam.detach().float().cpu().numpy()

    class_names = []
    class_ids = []
    cam_dict = {}
    for cam_idx, key in enumerate(keys):
        if key < 0 or key >= len(VOC_CLASS_NAMES):
            continue
        class_id = int(key) + 1
        class_names.append(VOC_CLASS_NAMES[int(key)])
        class_ids.append(class_id)
        cam_dict[class_id] = refined_cam[cam_idx]

    return class_names, class_ids, cam_dict


def generate_online_sgfr_labels(
    sgfr,
    inputs,
    refined_cam_infos,
    fallback_labels,
    use_box_prompt=True,
    use_point_prompt=False,
    ignore_index=255,
    return_raw_labels=False,
    joint_box_point_prompt=False,
    return_visualization=False,
):
    rgb_images = (
        denormalize_img(inputs.detach().cpu())
        .permute(0, 2, 3, 1)
        .numpy()
    )
    fallback_labels_cpu = fallback_labels.detach().cpu()
    pseudo_labels = []
    raw_labels = []
    valid_flags = []
    visualization_infos = []
    sgfr_count = 0
    accepted_fg_ratios = []

    for index, image in enumerate(rgb_images):
        fallback = fallback_labels_cpu[index].numpy().astype(np.uint8)
        pseudo_label = fallback
        raw_label = np.zeros_like(fallback, dtype=np.uint8)
        is_valid = False
        visualization_info = {'boxes': {}, 'points': {}}

        try:
            class_names, class_ids, cam_dict = (
                refined_cam_info_to_sgfr_inputs(refined_cam_infos[index])
            )
            if class_ids:
                generated_output = sgfr.generate(
                    image=image,
                    class_names=class_names,
                    class_ids=class_ids,
                    cam=cam_dict if use_point_prompt else None,
                    joint_box_point_prompt=joint_box_point_prompt,
                    use_box_prompts=use_box_prompt,
                    use_point_prompts=use_point_prompt,
                    ignore_index=ignore_index,
                    return_visualization=return_visualization,
                )
                if return_visualization:
                    generated, visualization_info = generated_output
                else:
                    generated = generated_output

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
                    is_valid = True
                    pseudo_label = generated
                    sgfr_count += 1
                    accepted_fg_ratios.append(fg_ratio)
        except Exception as error:
            warnings.warn(
                f"Online SGFR failed for sample {index}: {error}. "
                "Falling back to CAM label."
            )

        pseudo_labels.append(torch.from_numpy(pseudo_label.astype(np.int64)))
        raw_labels.append(torch.from_numpy(raw_label.astype(np.int64)))
        valid_flags.append(is_valid)
        if return_visualization:
            visualization_info['sam_pseudo_label'] = raw_label.copy()
            visualization_info['training_pseudo_label'] = pseudo_label.copy()
            visualization_info['used_cam_fallback'] = not is_valid
            visualization_infos.append(visualization_info)

    avg_fg_ratio = (
        float(np.mean(accepted_fg_ratios)) if accepted_fg_ratios else 0.0
    )
    pseudo_labels = torch.stack(pseudo_labels, dim=0).to(
        fallback_labels.device
    )
    if return_raw_labels:
        raw_labels = torch.stack(raw_labels, dim=0).to(
            fallback_labels.device
        )
        outputs = (
            pseudo_labels,
            sgfr_count,
            avg_fg_ratio,
            raw_labels,
            valid_flags,
        )
        return outputs + (
            (visualization_infos,) if return_visualization else ()
        )

    if return_visualization:
        return (
            pseudo_labels,
            sgfr_count,
            avg_fg_ratio,
            visualization_infos,
        )
    return pseudo_labels, sgfr_count, avg_fg_ratio
