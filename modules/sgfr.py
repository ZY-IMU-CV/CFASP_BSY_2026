"""
SGFR: SAM and Grounding-DINO Fusion Refinement Module

This module generates high-quality pseudo-labels by combining:
1. Grounding-DINO for open-set object detection with class name prompts
2. CAM-derived point prompts for comprehensive coverage
3. SAM 1 or SAM 3 PVS for precise segmentation with both box and point prompts
4. Conflict resolution strategy for aggregating class-specific masks

Reference: 
    "Integrating Foundation Models with Capsule Networks for Enhanced 
    Weakly-Supervised Semantic Segmentation" - ESWA 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from typing import List, Dict, Tuple, Optional, Union
import warnings
import os
from contextlib import nullcontext


class SGFR(nn.Module):
    """
    SAM and Grounding-DINO Fusion Refinement Module
    
    Args:
        sam_backend: Segmentation backend ('sam3' by default, or 'sam1')
        sam_checkpoint: Path to a SAM 1 checkpoint
        sam_model_type: SAM 1 model type ('vit_h', 'vit_l', 'vit_b')
        sam3_checkpoint: Path to the local SAM 3 checkpoint (offline mode)
        sam3_bpe_path: Optional path to the SAM 3 BPE vocabulary
        sam3_resolution: SAM 3 image encoder resolution
        sam3_multimask_output: Ask SAM 3 PVS for multiple candidate masks
        sam3_compile: Enable torch.compile in the official SAM 3 builder
        gdino_checkpoint: Path to Grounding-DINO checkpoint (optional)
        gdino_config: Path to Grounding-DINO config (optional)
        gdino_text_encoder_type: Grounding-DINO text encoder name or local path
        device: Device to run models on
        box_threshold: Confidence threshold for Grounding-DINO boxes
        text_threshold: Text confidence threshold for Grounding-DINO
        nms_threshold: NMS threshold for overlapping boxes
        cam_peak_threshold: Threshold for extracting CAM peaks as point prompts
        use_sam: Whether to use SAM for segmentation
        use_gdino: Whether to use Grounding-DINO for detection
    """
    
    def __init__(
        self,
        sam_checkpoint: str = None,
        sam_model_type: str = 'vit_h',
        sam_backend: str = 'sam3',
        sam3_checkpoint: str = None,
        sam3_bpe_path: str = None,
        sam3_resolution: int = 1008,
        sam3_multimask_output: bool = False,
        sam3_compile: bool = False,
        gdino_checkpoint: str = None,
        gdino_config: str = None,
        gdino_text_encoder_type: str = None,
        device: str = 'cuda',
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        nms_threshold: float = 0.5,
        cam_peak_threshold: float = 0.5,
        use_sam: bool = True,
        use_gdino: bool = True
    ):
        super().__init__()
        
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_threshold = nms_threshold
        self.cam_peak_threshold = cam_peak_threshold
        self.use_sam = use_sam
        self.use_gdino = use_gdino
        self.gdino_text_encoder_type = gdino_text_encoder_type
        self.sam_backend = str(sam_backend).lower()
        if self.sam_backend not in {'sam1', 'sam3'}:
            raise ValueError(
                f"Unsupported sam_backend={sam_backend!r}. Expected 'sam1' or 'sam3'."
            )
        self.sam3_multimask_output = bool(sam3_multimask_output)
        
        # Initialize the selected SAM visual-prompt backend.
        self.sam_predictor = None
        self.sam3_model = None
        self.sam3_processor = None
        if use_sam:
            if self.sam_backend == 'sam3':
                self._init_sam3(
                    checkpoint=sam3_checkpoint,
                    bpe_path=sam3_bpe_path,
                    resolution=sam3_resolution,
                    compile_model=sam3_compile
                )
            elif sam_checkpoint:
                self._init_sam1(sam_checkpoint, sam_model_type)
            else:
                warnings.warn("SAM 1 backend selected without sam_checkpoint. SAM features disabled.")
                self.use_sam = False
        
        # Initialize Grounding-DINO
        self.gdino_model = None
        if use_gdino and gdino_checkpoint:
            self._init_grounding_dino(gdino_checkpoint, gdino_config)
    
    @property
    def sam_is_ready(self) -> bool:
        """Whether the selected SAM backend was initialized successfully."""
        if not self.use_sam:
            return False
        if self.sam_backend == 'sam3':
            return self.sam3_model is not None and self.sam3_processor is not None
        return self.sam_predictor is not None

    def _sam3_autocast_context(self):
        """Keep SAM3 mixed precision local to SAM3 inference calls."""
        if torch.device(self.device).type == "cuda":
            return torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            )
        return nullcontext()

    def _init_sam1(self, checkpoint: str, model_type: str):
        """Initialize the original Segment Anything Model."""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(self.device)
            self.sam_predictor = SamPredictor(sam)
            print(f"SAM 1 ({model_type}) loaded successfully")
        except ImportError:
            warnings.warn("segment-anything not installed. SAM features disabled.")
            self.use_sam = False
        except Exception as e:
            warnings.warn(f"Failed to load SAM 1: {e}")
            self.use_sam = False

    def _init_sam3(
        self,
        checkpoint: str,
        bpe_path: str,
        resolution: int,
        compile_model: bool
    ):
        """Initialize official SAM 3 with its SAM-style PVS interaction head."""
        if not checkpoint:
            raise ValueError(
                "SAM 3 offline mode requires a local checkpoint path via "
                "sam3_checkpoint."
            )

        try:
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor

            self.sam3_model = build_sam3_image_model(
                bpe_path=bpe_path,
                device=self.device,
                eval_mode=True,
                checkpoint_path=checkpoint,
                load_from_HF=False,
                enable_inst_interactivity=True,
                compile=bool(compile_model)
            )
            self.sam3_processor = Sam3Processor(
                self.sam3_model,
                resolution=int(resolution),
                device=self.device
            )
            # Some SAM3 usage patterns enter CUDA autocast globally. Do not
            # allow that state to leak into WeCLIP or Grounding-DINO; SAM3
            # calls below establish their own scoped BF16 context instead.
            try:
                cuda_autocast_enabled = torch.is_autocast_enabled("cuda")
            except TypeError:
                cuda_autocast_enabled = torch.is_autocast_enabled()
            if cuda_autocast_enabled:
                try:
                    torch.set_autocast_enabled("cuda", False)
                except TypeError:
                    torch.set_autocast_enabled(False)
            print(f"SAM 3 PVS loaded successfully from local checkpoint {checkpoint}")
        except ImportError:
            warnings.warn(
                "sam3 is not installed. Install the official facebookresearch/sam3 "
                "package before using sam_backend='sam3'."
            )
            self.use_sam = False
        except Exception as e:
            warnings.warn(f"Failed to load SAM 3 PVS: {e}")
            self.use_sam = False
    
    def _init_grounding_dino(self, checkpoint: str, config: str):
        """Initialize Grounding-DINO model"""
        try:
            from groundingdino.models import build_model
            from groundingdino.util.misc import clean_state_dict
            from groundingdino.util.slconfig import SLConfig

            args = SLConfig.fromfile(config)
            args.device = self.device
            # Grounding-DINO is frozen inference in SGFR. Its deformable
            # attention CUDA extension does not support BF16, and activation
            # checkpointing is unnecessary when gradients are disabled.
            args.use_checkpoint = False
            if self.gdino_text_encoder_type:
                if os.path.isdir(self.gdino_text_encoder_type):
                    config_json = os.path.join(self.gdino_text_encoder_type, "config.json")
                    if not os.path.exists(config_json):
                        raise FileNotFoundError(
                            f"Grounding-DINO text encoder directory lacks config.json: {self.gdino_text_encoder_type}"
                        )
                args.text_encoder_type = self.gdino_text_encoder_type
                print(f"Override Grounding-DINO text_encoder_type: {args.text_encoder_type}")

            self.gdino_model = build_model(args)
            checkpoint_data = torch.load(checkpoint, map_location="cpu")
            self.gdino_model.load_state_dict(clean_state_dict(checkpoint_data["model"]), strict=False)
            self.gdino_model.eval()
            self.gdino_model.to(self.device).float()
            print("Grounding-DINO loaded successfully")
        except ImportError:
            warnings.warn("groundingdino not installed. Grounding-DINO features disabled.")
            self.use_gdino = False
        except Exception as e:
            warnings.warn(f"Failed to load Grounding-DINO: {e}")
            self.use_gdino = False
    
    def detect_with_gdino(
        self,
        image: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, List[np.ndarray]]:
        """
        Detect objects using Grounding-DINO
        
        Args:
            image: Input image (H, W, 3) in RGB format
            class_names: List of class names to detect
            
        Returns:
            Dictionary mapping class names to list of bounding boxes
        """
        if not self.use_gdino or self.gdino_model is None:
            return {name: [] for name in class_names}
        
        try:
            from groundingdino.datasets import transforms as T
            from groundingdino.util.inference import predict
            from torchvision.ops import box_convert
            
            prompt_names = [self._to_gdino_prompt_name(name) for name in class_names]
            caption = " . ".join(prompt_names) + " ."

            image = image.astype(np.uint8)
            image_pil = Image.fromarray(image).convert("RGB")
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            image_tensor, _ = transform(image_pil, None)
            
            gdino_autocast = (
                torch.autocast(device_type="cuda", enabled=False)
                if torch.device(self.device).type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), gdino_autocast:
                boxes, logits, phrases = predict(
                    model=self.gdino_model,
                    image=image_tensor.float(),
                    caption=caption,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )
            
            result = {name: [] for name in class_names}
            if len(boxes) == 0:
                return result

            h, w = image.shape[:2]
            boxes = boxes.detach().cpu() * torch.tensor([w, h, w, h], dtype=boxes.dtype)
            boxes = box_convert(
                boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy"
            ).float().numpy()
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)

            for box, phrase in zip(boxes, phrases):
                for class_name in class_names:
                    if self._phrase_matches_class(phrase, class_name):
                        result[class_name].append(box.astype(np.float32))
                        break
            
            return result
            
        except Exception as e:
            warnings.warn(f"Grounding-DINO detection failed: {e}")
            return {name: [] for name in class_names}

    @staticmethod
    def _normalize_prompt_text(text: str) -> str:
        return text.lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _to_gdino_prompt_name(class_name: str) -> str:
        prompt_names = {
            "aeroplane": "airplane",
            "motorbike": "motorcycle",
            "diningtable": "dining table",
            "pottedplant": "potted plant",
            "tvmonitor": "tv monitor",
        }
        return prompt_names.get(class_name.lower(), class_name)

    def _phrase_matches_class(self, phrase: str, class_name: str) -> bool:
        aliases = {
            "aeroplane": ["aeroplane", "airplane", "aircraft", "plane"],
            "motorbike": ["motorbike", "motorcycle"],
            "diningtable": ["diningtable", "dining table"],
            "pottedplant": ["pottedplant", "potted plant", "plant"],
            "tvmonitor": ["tvmonitor", "tv monitor", "television", "tv", "screen"],
            "sofa": ["sofa", "couch"],
            "person": ["person", "people", "human", "man", "woman"],
        }
        phrase_norm = self._normalize_prompt_text(phrase)
        candidates = aliases.get(class_name.lower(), [class_name])
        return any(self._normalize_prompt_text(candidate) in phrase_norm for candidate in candidates)
    
    def extract_cam_points(
        self,
        cam: np.ndarray,
        threshold: float = None,
        max_points: int = 5,
        box: np.ndarray = None
    ) -> List[Tuple[int, int]]:
        """
        Extract salient point prompts from CAM local peaks
        
        Args:
            cam: Class Activation Map (H, W)
            threshold: Activation threshold
            max_points: Maximum number of points to extract
            box: Optional [x1, y1, x2, y2] region. When provided, only
                peaks inside the box are returned.
            
        Returns:
            List of (x, y) coordinates
        """
        if threshold is None:
            threshold = self.cam_peak_threshold
        
        # Normalize CAM
        cam = cam.astype(np.float32)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Find local maxima
        from scipy.ndimage import maximum_filter, label
        
        local_max = maximum_filter(cam, size=20) == cam
        threshold_mask = cam > threshold
        valid_region = np.ones_like(cam, dtype=bool)
        if box is not None:
            h, w = cam.shape[:2]
            x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
            x1 = int(np.clip(np.floor(x1), 0, w - 1))
            y1 = int(np.clip(np.floor(y1), 0, h - 1))
            x2 = int(np.clip(np.ceil(x2), 0, w - 1))
            y2 = int(np.clip(np.ceil(y2), 0, h - 1))
            valid_region.fill(False)
            if x2 >= x1 and y2 >= y1:
                valid_region[y1:y2 + 1, x1:x2 + 1] = True

        peaks = local_max & threshold_mask & valid_region
        
        # Get peak coordinates
        y_coords, x_coords = np.where(peaks)
        
        if len(x_coords) == 0:
            # Fall back to the strongest CAM response in the valid region.
            candidate_cam = np.where(valid_region, cam, -np.inf)
            if not np.isfinite(candidate_cam).any():
                return []
            y, x = np.unravel_index(candidate_cam.argmax(), candidate_cam.shape)
            return [(int(x), int(y))]
        
        # Sort by activation value and take top-k
        values = cam[y_coords, x_coords]
        sorted_indices = np.argsort(values)[::-1][:max_points]
        
        points = [(int(x_coords[i]), int(y_coords[i])) for i in sorted_indices]
        return points
    
    def segment_with_sam(
        self,
        image: np.ndarray,
        boxes: List[np.ndarray] = None,
        points: List[Tuple[int, int]] = None,
        point_labels: List[int] = None,
        points_per_box: List[List[Tuple[int, int]]] = None,
        inference_state=None
    ) -> List[np.ndarray]:
        """
        Segment objects using SAM with box and/or point prompts
        
        Args:
            image: Input image (H, W, 3) in RGB format
            boxes: List of bounding boxes [x1, y1, x2, y2]
            points: List of (x, y) point coordinates
            point_labels: Labels for points (1=foreground, 0=background)
            points_per_box: Optional point prompts corresponding to each box.
                When supplied, every SAM prediction receives its box and its
                corresponding points in the same call.
            inference_state: Prepared per-image SAM state. Passing it avoids
                recomputing image features for every class in the image.
            
        Returns:
            List of binary masks
        """
        if not self.sam_is_ready:
            return []
        
        try:
            if inference_state is None:
                inference_state = self._prepare_sam_image(image)
            
            masks = []
            
            # Segment with boxes
            if boxes:
                for box_idx, box in enumerate(boxes):
                    box_array = np.asarray(box, dtype=np.float32)
                    box_points = points
                    if points_per_box is not None:
                        box_points = points_per_box[box_idx]

                    point_coords, labels = None, None
                    if box_points:
                        point_coords = np.asarray(box_points, dtype=np.float32)
                        if point_labels is None or len(point_labels) != len(box_points):
                            labels = np.ones(len(box_points), dtype=np.int32)
                        else:
                            labels = np.asarray(point_labels, dtype=np.int32)

                    masks.append(
                        self._predict_sam_mask(
                            inference_state=inference_state,
                            box=box_array,
                            point_coords=point_coords,
                            point_labels=labels
                        )
                    )
            
            # Segment with points
            if points and not boxes:
                point_coords = np.asarray(points, dtype=np.float32)
                if point_labels is None:
                    point_labels = np.ones(len(points), dtype=np.int32)
                else:
                    point_labels = np.asarray(point_labels, dtype=np.int32)
                
                masks.append(
                    self._predict_sam_mask(
                        inference_state=inference_state,
                        point_coords=point_coords,
                        point_labels=point_labels
                    )
                )
            
            return masks
            
        except Exception as e:
            warnings.warn(f"SAM segmentation failed: {e}")
            return []

    def _prepare_sam_image(self, image: np.ndarray):
        """Encode one image once for all class-wise visual prompt calls."""
        image = np.clip(image, 0, 255).astype(np.uint8)
        if self.sam_backend == 'sam3':
            image_pil = Image.fromarray(image).convert("RGB")
            with torch.inference_mode(), self._sam3_autocast_context():
                return self.sam3_processor.set_image(image_pil)

        self.sam_predictor.set_image(image)
        return True

    @staticmethod
    def _to_numpy(value):
        if torch.is_tensor(value):
            value = value.detach().cpu()
            # SAM3 may return masks/scores in bfloat16 under mixed precision,
            # while NumPy has no native bfloat16 representation.
            if value.dtype == torch.bfloat16:
                value = value.float()
            return value.numpy()
        return np.asarray(value)

    def _predict_sam_mask(
        self,
        inference_state,
        box: np.ndarray = None,
        point_coords: np.ndarray = None,
        point_labels: np.ndarray = None
    ) -> np.ndarray:
        """Run one joint visual-prompt prediction on the selected backend."""
        if self.sam_backend == 'sam3':
            with torch.inference_mode(), self._sam3_autocast_context():
                predicted_masks, scores, _ = self.sam3_model.predict_inst(
                    inference_state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=self.sam3_multimask_output
                )
        else:
            predicted_masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=False
            )

        predicted_masks = self._to_numpy(predicted_masks)
        if predicted_masks.ndim == 2:
            return predicted_masks

        scores = self._to_numpy(scores).reshape(-1)
        best_idx = int(np.argmax(scores)) if scores.size else 0
        return predicted_masks[best_idx]
    
    def resolve_conflicts(
        self,
        masks: Dict[int, List[np.ndarray]],
        h: int,
        w: int,
        ignore_index: int = 255
    ) -> np.ndarray:
        """
        Resolve conflicts between overlapping masks from different classes
        
        Strategy:
        - For overlapping regions, assign to the class with smaller mask area
          (assuming smaller objects are more likely to be occluded)
        - Background is assigned where no object mask exists
        
        Args:
            masks: Dictionary mapping class_id to list of masks
            h, w: Output dimensions
            ignore_index: Index for ignored regions
            
        Returns:
            Unified pseudo-label map (H, W)
        """
        # Initialize with background
        pseudo_label = np.zeros((h, w), dtype=np.uint8)
        
        # Track mask areas for conflict resolution
        area_map = np.full((h, w), np.inf)
        
        for class_id, class_masks in masks.items():
            if class_id == 0:  # Skip background
                continue
                
            for mask in class_masks:
                # Resize mask if needed
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype(np.uint8), (w, h), 
                                     interpolation=cv2.INTER_NEAREST)
                
                mask_bool = mask > 0
                mask_area = mask_bool.sum()
                
                # Update pixels where this mask has smaller area (conflict resolution)
                update_mask = mask_bool & (mask_area < area_map)
                pseudo_label[update_mask] = class_id
                area_map[update_mask] = mask_area
        
        return pseudo_label
    
    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor, Image.Image],
        class_names: List[str],
        class_ids: List[int] = None,
        cam: np.ndarray = None,
        ignore_index: int = 255,
        joint_box_point_prompt: bool = False,
        use_box_prompts: bool = True,
        use_point_prompts: bool = True,
        return_visualization: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate high-quality pseudo-label for a single image
        
        Args:
            image: Input image
            class_names: List of class names present in the image
            class_ids: Corresponding class IDs (if None, uses 1-indexed)
            cam: Optional CAM for point prompt extraction (H, W, num_classes) or dict
            ignore_index: Index for ignored regions
            joint_box_point_prompt: If True, each SAM call receives a DINO box
                and the CAM points inside that box together. If False, box- and
                point-prompt masks are generated separately and then aggregated.
            use_box_prompts: Whether to run Grounding-DINO and send its boxes
                to SAM.
            use_point_prompts: Whether to extract CAM peaks and send them to
                SAM. With boxes disabled these are global CAM peaks.
            return_visualization: Return the actual class-wise boxes and points
                used by SAM together with the pseudo-label.
            
        Returns:
            Pseudo-label map (H, W)
        """
        # Convert image to numpy RGB
        if isinstance(image, torch.Tensor):
            image = image.detach().float().cpu().numpy()
            if image.ndim == 4:
                image = image[0]
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
            # Denormalize if needed
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
        elif isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        
        if class_ids is None:
            class_ids = list(range(1, len(class_names) + 1))

        use_box_prompts = bool(use_box_prompts)
        use_point_prompts = bool(use_point_prompts)
        joint_box_point_prompt = bool(joint_box_point_prompt)
        if joint_box_point_prompt and not (use_box_prompts and use_point_prompts):
            raise ValueError(
                "joint_box_point_prompt requires both box and point prompts"
            )
        if not use_point_prompts:
            cam = None
        
        # Storage for all masks
        all_masks = {class_id: [] for class_id in class_ids}
        visualization_info = {
            'boxes': {int(class_id): [] for class_id in class_ids},
            'points': {int(class_id): [] for class_id in class_ids},
        }
        
        # Step 1: Detect with Grounding-DINO
        detections = {name: [] for name in class_names}
        if use_box_prompts and self.use_gdino:
            detections = self.detect_with_gdino(image, class_names)
        if return_visualization:
            for class_idx, class_name in enumerate(class_names):
                class_id = int(class_ids[class_idx])
                visualization_info['boxes'][class_id] = [
                    np.asarray(box, dtype=np.float32).tolist()
                    for box in detections.get(class_name, [])
                ]

        has_box_prompts = (
            use_box_prompts
            and any(bool(boxes) for boxes in detections.values())
        )
        needs_point_only_prompt = (
            use_point_prompts
            and cam is not None
            and not joint_box_point_prompt
        )
        sam_inference_state = None
        sam_ready_for_image = (
            self.sam_is_ready and (has_box_prompts or needs_point_only_prompt)
        )
        if sam_ready_for_image:
            try:
                sam_inference_state = self._prepare_sam_image(image)
            except Exception as e:
                warnings.warn(f"Failed to encode image with {self.sam_backend}: {e}")
                sam_ready_for_image = False

        if (
            joint_box_point_prompt
            and use_box_prompts
            and use_point_prompts
            and cam is not None
            and sam_ready_for_image
        ):
            # Joint mode: every mask is produced by one SAM prediction that
            # receives a DINO box and CAM-derived points at the same time.
            for i, class_name in enumerate(class_names):
                class_id = class_ids[i]
                boxes = detections.get(class_name, [])
                if not boxes:
                    continue

                if isinstance(cam, dict):
                    class_cam = cam.get(class_name, cam.get(class_id))
                elif cam.ndim == 3:
                    class_cam = cam[:, :, i]
                else:
                    class_cam = cam

                if class_cam is None:
                    continue
                if class_cam.shape[:2] != (h, w):
                    class_cam = cv2.resize(
                        class_cam.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_LINEAR
                    )

                joint_pairs = [
                    (box, self.extract_cam_points(class_cam, box=box))
                    for box in boxes
                ]
                joint_pairs = [(box, points) for box, points in joint_pairs if points]
                if not joint_pairs:
                    continue

                joint_boxes = [box for box, _ in joint_pairs]
                points_per_box = [points for _, points in joint_pairs]
                if return_visualization:
                    visualization_info['points'][int(class_id)] = [
                        [int(x), int(y)]
                        for points in points_per_box
                        for x, y in points
                    ]
                masks = self.segment_with_sam(
                    image,
                    boxes=joint_boxes,
                    points_per_box=points_per_box,
                    inference_state=sam_inference_state
                )
                all_masks[class_id].extend(masks)
        else:
            # Legacy modes: box-only, or separate box/point masks followed by
            # aggregation when CAM points are enabled.
            if use_box_prompts:
                for class_name, boxes in detections.items():
                    class_idx = class_names.index(class_name)
                    class_id = class_ids[class_idx]
                    
                    if boxes and sam_ready_for_image:
                        masks = self.segment_with_sam(
                            image,
                            boxes=boxes,
                            inference_state=sam_inference_state
                        )
                        all_masks[class_id].extend(masks)

            # Step 3: Extract point prompts from CAM and segment separately.
            if use_point_prompts and cam is not None and sam_ready_for_image:
                for i, class_name in enumerate(class_names):
                    class_id = class_ids[i]
                    
                    # Get CAM for this class
                    if isinstance(cam, dict):
                        class_cam = cam.get(class_name, cam.get(class_id))
                    elif cam.ndim == 3:
                        class_cam = cam[:, :, i]
                    else:
                        class_cam = cam
                    
                    if class_cam is not None:
                        # Extract point prompts
                        points = self.extract_cam_points(class_cam)
                        
                        if points:
                            if return_visualization:
                                visualization_info['points'][int(class_id)] = [
                                    [int(x), int(y)] for x, y in points
                                ]
                            # Segment with points
                            point_masks = self.segment_with_sam(
                                image, 
                                points=points,
                                point_labels=[1] * len(points),
                                inference_state=sam_inference_state
                            )
                            all_masks[class_id].extend(point_masks)
        
        # Step 4: Resolve conflicts and create unified pseudo-label
        pseudo_label = self.resolve_conflicts(all_masks, h, w, ignore_index)
        if return_visualization:
            visualization_info['pseudo_label'] = pseudo_label.copy()
            return pseudo_label, visualization_info
        return pseudo_label
    
    def generate_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        class_names_list: List[List[str]],
        class_ids_list: List[List[int]] = None,
        cams: List[np.ndarray] = None,
        ignore_index: int = 255
    ) -> List[np.ndarray]:
        """
        Generate pseudo-labels for a batch of images
        
        Args:
            images: List of input images
            class_names_list: List of class name lists for each image
            class_ids_list: List of class ID lists for each image
            cams: List of CAMs for each image
            ignore_index: Index for ignored regions
            
        Returns:
            List of pseudo-label maps
        """
        pseudo_labels = []
        
        for i, image in enumerate(images):
            class_names = class_names_list[i]
            class_ids = class_ids_list[i] if class_ids_list else None
            cam = cams[i] if cams else None
            
            pseudo_label = self.generate(
                image=image,
                class_names=class_names,
                class_ids=class_ids,
                cam=cam,
                ignore_index=ignore_index
            )
            pseudo_labels.append(pseudo_label)
        
        return pseudo_labels


class SimpleSGFR(nn.Module):
    """
    Simplified SGFR module without external dependencies
    Uses CAM refinement with morphological operations as fallback
    """
    
    def __init__(
        self,
        cam_threshold: float = 0.5,
        bg_threshold: float = 0.1,
        kernel_size: int = 5
    ):
        super().__init__()
        self.cam_threshold = cam_threshold
        self.bg_threshold = bg_threshold
        self.kernel_size = kernel_size
    
    def refine_cam(
        self,
        cam: np.ndarray,
        fg_threshold: float = None,
        bg_threshold: float = None
    ) -> np.ndarray:
        """
        Refine CAM using morphological operations
        
        Args:
            cam: Class Activation Map (H, W) or (H, W, C)
            fg_threshold: Foreground threshold
            bg_threshold: Background threshold
            
        Returns:
            Refined binary mask
        """
        if fg_threshold is None:
            fg_threshold = self.cam_threshold
        if bg_threshold is None:
            bg_threshold = self.bg_threshold
        
        # Normalize
        cam = cam.astype(np.float32)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        # Create initial mask
        fg_mask = (cam > fg_threshold).astype(np.uint8)
        
        # Morphological refinement
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.kernel_size, self.kernel_size)
        )
        
        # Close small holes
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        return fg_mask
    
    def generate(
        self,
        cams: np.ndarray,
        class_ids: List[int],
        ignore_index: int = 255
    ) -> np.ndarray:
        """
        Generate pseudo-label from multi-class CAMs
        
        Args:
            cams: Multi-class CAMs (H, W, num_classes) or dict
            class_ids: List of class IDs
            ignore_index: Index for ignored/uncertain regions
            
        Returns:
            Pseudo-label map (H, W)
        """
        if isinstance(cams, dict):
            h, w = list(cams.values())[0].shape[:2]
        else:
            h, w = cams.shape[:2]
        
        # Initialize with background
        pseudo_label = np.zeros((h, w), dtype=np.uint8)
        confidence = np.zeros((h, w), dtype=np.float32)
        
        for i, class_id in enumerate(class_ids):
            if isinstance(cams, dict):
                cam = cams[class_id]
            else:
                cam = cams[:, :, i]
            
            # Refine CAM
            mask = self.refine_cam(cam)
            
            # Update pseudo-label where confidence is higher
            cam_norm = cam.astype(np.float32)
            if cam_norm.max() > cam_norm.min():
                cam_norm = (cam_norm - cam_norm.min()) / (cam_norm.max() - cam_norm.min())
            
            update_mask = (mask > 0) & (cam_norm > confidence)
            pseudo_label[update_mask] = class_id
            confidence[update_mask] = cam_norm[update_mask]
        
        # Mark low-confidence regions as ignore
        uncertain_mask = (confidence > self.bg_threshold) & (confidence < self.cam_threshold)
        # pseudo_label[uncertain_mask] = ignore_index  # Optional: mark uncertain regions
        
        return pseudo_label


def create_sgfr(
    sam_checkpoint: str = None,
    gdino_checkpoint: str = None,
    gdino_config: str = None,
    gdino_text_encoder_type: str = None,
    device: str = 'cuda',
    use_sam: bool = True,
    use_gdino: bool = True,
    **kwargs
) -> SGFR:
    """
    Factory function to create SGFR module
    Falls back to SimpleSGFR if dependencies are missing
    """
    if use_sam or use_gdino:
        try:
            return SGFR(
                sam_checkpoint=sam_checkpoint,
                gdino_checkpoint=gdino_checkpoint,
                gdino_config=gdino_config,
                gdino_text_encoder_type=gdino_text_encoder_type,
                device=device,
                use_sam=use_sam,
                use_gdino=use_gdino,
                **kwargs
            )
        except Exception as e:
            warnings.warn(f"Failed to create SGFR with SAM/GDINO: {e}. Using SimpleSGFR.")
    
    return SimpleSGFR(**kwargs)
