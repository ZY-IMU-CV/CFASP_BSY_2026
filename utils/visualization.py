import json
import os
import re
import shutil

import cv2
import numpy as np


def _to_numpy(value):
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value)


def _safe_name(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def _voc_colormap(size=256):
    cmap = np.zeros((size, 3), dtype=np.uint8)
    for index in range(size):
        value = index
        red = green = blue = 0
        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        cmap[index] = [red, green, blue]
    return cmap


VOC_COLORMAP = _voc_colormap()


def _write_rgb(path, image):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR))


def _resize_map(array, height, width, interpolation):
    if array.shape[:2] == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _heatmap_overlay(image, activation, alpha):
    height, width = image.shape[:2]
    activation = _resize_map(
        _to_numpy(activation).astype(np.float32),
        height,
        width,
        cv2.INTER_LINEAR,
    )
    finite = np.isfinite(activation)
    if finite.any():
        minimum = float(activation[finite].min())
        maximum = float(activation[finite].max())
        if maximum > minimum:
            activation = (activation - minimum) / (maximum - minimum)
        else:
            activation = np.zeros_like(activation)
    activation = np.clip(activation, 0.0, 1.0)
    heatmap_bgr = cv2.applyColorMap(
        np.uint8(np.round(activation * 255.0)), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image, 1.0 - alpha, heatmap, alpha, 0.0)


def _colorize_label(label, height, width):
    label = _resize_map(
        _to_numpy(label).astype(np.uint8),
        height,
        width,
        cv2.INTER_NEAREST,
    )
    return VOC_COLORMAP[label]


def _logits_to_label(logits, height, width):
    logits = _to_numpy(logits).astype(np.float32)
    resized = np.stack(
        [
            cv2.resize(channel, (width, height), interpolation=cv2.INTER_LINEAR)
            for channel in logits
        ],
        axis=0,
    )
    return resized.argmax(axis=0).astype(np.uint8)


def _draw_boxes(image, boxes_by_class, class_names):
    output = image.copy()
    for class_id, boxes in (boxes_by_class or {}).items():
        class_id = int(class_id)
        color = tuple(int(v) for v in VOC_COLORMAP[class_id % 256])
        class_name = (
            class_names[class_id - 1]
            if 0 < class_id <= len(class_names)
            else str(class_id)
        )
        for box in boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                output,
                class_name,
                (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
    return output


def _draw_points(image, points_by_class):
    output = image.copy()
    for class_id, points in (points_by_class or {}).items():
        class_id = int(class_id)
        color = tuple(int(v) for v in VOC_COLORMAP[class_id % 256])
        for x, y in points:
            center = (int(round(x)), int(round(y)))
            cv2.circle(output, center, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(output, center, 5, color, -1, cv2.LINE_AA)
    return output


def _with_title(image, title):
    title_height = 34
    canvas = np.full(
        (image.shape[0] + title_height, image.shape[1], 3), 255,
        dtype=np.uint8,
    )
    canvas[title_height:] = image
    cv2.putText(
        canvas,
        title,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _make_panel(tiles, columns=3):
    titled = [_with_title(image, title) for title, image in tiles]
    rows = []
    for start in range(0, len(titled), columns):
        row = titled[start:start + columns]
        while len(row) < columns:
            row.append(np.full_like(titled[0], 255))
        rows.append(np.concatenate(row, axis=1))
    return np.concatenate(rows, axis=0)


def save_voc_validation_visualization(
    output_root,
    sample_name,
    image,
    cam_info,
    prompt_info,
    seg_clip,
    seg_dino,
    seg_fused,
    class_names,
    overlay_alpha=0.55,
):
    """Save paper-ready intermediate outputs and a compact 3x3 panel."""
    sample_dir = os.path.join(output_root, _safe_name(sample_name))
    os.makedirs(sample_dir, exist_ok=True)
    for legacy_overlay in (
        'sam3_pseudo_overlay.png',
        'clip_segmentation_overlay.png',
        'dino_segmentation_overlay.png',
        'fused_segmentation_overlay.png',
        'sam3_pseudo_label_cmap.png',
        'clip_segmentation_cmap.png',
        'dino_segmentation_cmap.png',
        'fused_segmentation_cmap.png',
    ):
        legacy_path = os.path.join(sample_dir, legacy_overlay)
        if os.path.isfile(legacy_path):
            os.remove(legacy_path)
    legacy_cam_dir = os.path.join(sample_dir, 'cams')
    if os.path.isdir(legacy_cam_dir):
        shutil.rmtree(legacy_cam_dir)

    image = np.clip(_to_numpy(image), 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    keys = [int(key) for key in np.asarray(cam_info['keys']).tolist()]
    initial_cams = _to_numpy(cam_info['initial_cam'])
    refined_cams = _to_numpy(cam_info['refined_cam'])

    if keys:
        target_index = int(
            np.argmax(refined_cams.reshape(len(keys), -1).mean(axis=1))
        )
        target_key = keys[target_index]
        target_name = class_names[target_key]
        initial_overlay = _heatmap_overlay(
            image, initial_cams[target_index], overlay_alpha
        )
        refined_overlay = _heatmap_overlay(
            image, refined_cams[target_index], overlay_alpha
        )
    else:
        target_key = None
        target_name = 'no_foreground_class'
        initial_overlay = image.copy()
        refined_overlay = image.copy()

    boxes = prompt_info.get('boxes', {}) if prompt_info else {}
    points = prompt_info.get('points', {}) if prompt_info else {}
    box_prompt = _draw_boxes(image, boxes, class_names)
    point_prompt = _draw_points(image, points)

    sam_label = (
        prompt_info.get('sam_pseudo_label')
        if prompt_info and prompt_info.get('sam_pseudo_label') is not None
        else np.zeros((height, width), dtype=np.uint8)
    )
    sam_label = _resize_map(
        _to_numpy(sam_label).astype(np.uint8),
        height,
        width,
        cv2.INTER_NEAREST,
    )
    sam_color = _colorize_label(sam_label, height, width)

    clip_label = _logits_to_label(seg_clip, height, width)
    clip_color = _colorize_label(clip_label, height, width)
    if seg_dino is not None:
        dino_label = _logits_to_label(seg_dino, height, width)
        dino_color = _colorize_label(dino_label, height, width)
    else:
        dino_label = np.zeros((height, width), dtype=np.uint8)
        dino_color = _colorize_label(dino_label, height, width)
    fused_label = _logits_to_label(seg_fused, height, width)
    fused_color = _colorize_label(fused_label, height, width)

    individual_images = {
        'input.png': image,
        'initial_cam.png': initial_overlay,
        'refined_cam.png': refined_overlay,
        'box_prompt.png': box_prompt,
        'point_prompt.png': point_prompt,
        'sam3_pseudo_label.png': sam_color,
        'clip_segmentation.png': clip_color,
        'dino_segmentation.png': dino_color,
        'fused_segmentation.png': fused_color,
    }
    for filename, output_image in individual_images.items():
        _write_rgb(os.path.join(sample_dir, filename), output_image)

    panel = _make_panel(
        [
            ('Input image', image),
            (f'Initial CAM: {target_name}', initial_overlay),
            (f'Refined CAM: {target_name}', refined_overlay),
            ('Box prompt', box_prompt),
            ('Point prompt', point_prompt),
            ('SAM3 pseudo-label', sam_color),
            ('CLIP prediction', clip_color),
            (
                'DINO prediction' if seg_dino is not None else 'DINO disabled',
                dino_color,
            ),
            ('Fused prediction', fused_color),
        ]
    )
    panel_path = os.path.join(sample_dir, 'framework_panel.png')
    _write_rgb(panel_path, panel)

    metadata = {
        'sample_name': str(sample_name),
        'target_cam_class_id': None if target_key is None else target_key + 1,
        'target_cam_class_name': target_name,
        'present_class_ids': [key + 1 for key in keys],
        'boxes': boxes,
        'points': points,
        'used_cam_fallback': bool(
            prompt_info.get('used_cam_fallback', True) if prompt_info else True
        ),
    }
    with open(os.path.join(sample_dir, 'metadata.json'), 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return panel_path
