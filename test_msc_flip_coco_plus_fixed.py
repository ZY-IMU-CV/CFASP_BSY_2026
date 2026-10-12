import argparse
import os
import sys

sys.path.append(".")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import imageio.v2 as imageio
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm

from datasets import coco
from utils import evaluate
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
from WeCLIP_Plus.model_attn_aff_coco import WeCLIP_Plus


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="configs/coco_attn_reg.yaml",
    type=str,
    help="config",
)
parser.add_argument(
    "--work_dir",
    default="results_coco",
    type=str,
    help="work_dir",
)
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set")
parser.add_argument(
    "--model_path",
    default=(
        "/home/stu_24/bsy/WeCLIP+/work_dir_coco/checkpoints/"
        "2025-05-28-23-30/wetr_iter_80000.pth"
    ),
    type=str,
    help="model_path",
)
parser.add_argument(
    "--scales",
    default="1.0,1.5",
    type=str,
    help="multi-scale factors, comma separated",
)
parser.add_argument(
    "--crf",
    action="store_true",
    default=True,
    help="run DenseCRF after model validation",
)
parser.add_argument(
    "--no_crf",
    action="store_false",
    dest="crf",
    help="skip DenseCRF after model validation",
)
parser.add_argument(
    "--only_crf",
    action="store_true",
    help="skip model inference and run DenseCRF from saved logits",
)
parser.add_argument(
    "--crf_jobs",
    default=0,
    type=int,
    help="number of CRF workers; 0 uses half of the CPU cores",
)
parser.add_argument(
    "--crf_batch_size",
    default=256,
    type=int,
    help="number of CRF results accumulated before metric update",
)


def parse_scales(scales):
    parsed = [float(scale) for scale in scales.split(",") if scale.strip()]
    if not parsed:
        raise ValueError("--scales must contain at least one valid scale")
    return parsed


def validate(model, dataset, cfg, test_scales=None):
    preds_buffer, gts_buffer, msc_preds_buffer, cams_buffer = [], [], [], []

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    model.cuda(0)
    model.eval()

    num = 0
    num_classes = int(cfg.dataset.num_classes)
    preds_hist = np.zeros((num_classes, num_classes))
    msc_preds_hist = np.zeros((num_classes, num_classes))
    cams_hist = np.zeros((num_classes, num_classes))

    for _, data in tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        ncols=100,
        ascii=" >=",
    ):
        num += 1
        name, inputs, labels, cls_labels = data
        names = name + name

        inputs = inputs.cuda()
        labels = labels.cuda()

        _, _, height, width = inputs.shape
        ratio = cfg.clip_init.resize_long / max(height, width)
        resized_height = int(height * ratio)
        resized_width = int(width * ratio)
        inputs = F.interpolate(
            inputs,
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_clip_cat, segs_dino_cat = model(inputs_cat, names, mode="val")
        segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat
        segs = segs_cat[0].unsqueeze(0)

        flipped_average = (
            segs_cat[0, ...] + segs_cat[1, ...].flip(-1)
        ) / 2
        segs_list.append(flipped_average)
        _, _, output_height, output_width = segs_cat.shape

        for scale in test_scales:
            if scale == 1.0:
                continue

            scaled_inputs = F.interpolate(
                inputs,
                scale_factor=scale,
                mode="bilinear",
                align_corners=False,
            )
            inputs_cat = torch.cat(
                [scaled_inputs, scaled_inputs.flip(-1)],
                dim=0,
            )
            segs_clip_cat, segs_dino_cat = model(
                inputs_cat,
                names,
                mode="val",
            )
            segs_cat = 0.5 * segs_clip_cat + 0.5 * segs_dino_cat
            segs_cat = F.interpolate(
                segs_cat,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )
            flipped_average = (
                segs_cat[0, ...] + segs_cat[1, ...].flip(-1)
            ) / 2
            segs_list.append(flipped_average)

        msc_segs = torch.mean(
            torch.stack(segs_list, dim=0),
            dim=0,
        ).unsqueeze(0)

        resized_segs = F.interpolate(
            segs,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        seg_preds = torch.argmax(resized_segs, dim=1)
        resized_msc_segs = F.interpolate(
            msc_segs,
            size=labels.shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        # The original script called this buffer "cams", but it actually stores
        # the multi-scale segmentation prediction. It is kept for output
        # compatibility with the original validation code.
        cams_buffer += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        preds_buffer += list(seg_preds.cpu().numpy().astype(np.int16))
        msc_preds_buffer += list(
            msc_seg_preds.cpu().numpy().astype(np.int16)
        )
        gts_buffer += list(labels.cpu().numpy().astype(np.int16))

        if num % 2000 == 0:
            preds_hist, seg_score = evaluate.scores(
                gts_buffer,
                preds_buffer,
                preds_hist,
                num_classes,
            )
            msc_preds_hist, msc_seg_score = evaluate.scores(
                gts_buffer,
                msc_preds_buffer,
                msc_preds_hist,
                num_classes,
            )
            cams_hist, cam_score = evaluate.scores(
                gts_buffer,
                cams_buffer,
                cams_hist,
                num_classes,
            )
            preds_buffer, gts_buffer = [], []
            msc_preds_buffer, cams_buffer = [], []
            print("cams score:")
            print(cam_score)
            print("segs score:")
            print(seg_score)
            print("msc segs score:")
            print(msc_seg_score)

        np.save(
            os.path.join(args.work_dir, "logit", name[0] + ".npy"),
            {
                "segs": segs.detach().cpu().numpy(),
                "msc_segs": msc_segs.detach().cpu().numpy(),
            },
        )

    return (
        gts_buffer,
        preds_buffer,
        msc_preds_buffer,
        cams_buffer,
        preds_hist,
        msc_preds_hist,
        cams_hist,
    )


def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(
        config.dataset.name_list_dir,
        args.eval_set,
    ) + ".txt"
    with open(txt_name) as file:
        name_list = [line for line in file.read().split("\n") if line]

    logit_dir = os.path.join(args.work_dir, "logit")
    missing_logits = [
        name
        for name in name_list
        if not os.path.isfile(os.path.join(logit_dir, name + ".npy"))
    ]
    if missing_logits:
        examples = ", ".join(missing_logits[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_logits)} saved logit files in "
            f"{logit_dir}. Examples: {examples}"
        )

    images_path = os.path.join(
        config.dataset.root_dir,
        "JPEGImages",
        "val2014",
    )
    labels_path = os.path.join(
        config.dataset.root_dir,
        "SegmentationClass",
        "val2014",
    )

    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    def _job(index):
        name = name_list[index]
        logit_name = os.path.join(logit_dir, name + ".npy")
        logit = np.load(logit_name, allow_pickle=True).item()["msc_segs"]

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)

        label_name = os.path.join(labels_path, name + ".png")
        label = imageio.imread(label_name)

        height, width, _ = image.shape
        logit = torch.as_tensor(logit, dtype=torch.float32)
        logit = F.interpolate(
            logit,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        prob = F.softmax(logit, dim=1)[0].numpy()
        prob = post_processor(image.astype(np.uint8), prob)
        pred = np.argmax(prob, axis=0).astype(np.uint8)

        imageio.imsave(
            os.path.join(args.work_dir, "prediction", name + ".png"),
            pred,
        )
        imageio.imsave(
            os.path.join(
                args.work_dir,
                "prediction_cmap",
                name + ".png",
            ),
            encode_cmap(pred).astype(np.uint8),
        )
        return pred, label

    n_jobs = (
        args.crf_jobs
        if args.crf_jobs > 0
        else max(1, int(multiprocessing.cpu_count() * 0.5))
    )
    batch_size = max(1, args.crf_batch_size)
    num_classes = int(config.dataset.num_classes)
    hist = np.zeros((num_classes, num_classes))

    # Do not retain all 40k full-resolution masks in memory at the same time.
    with joblib.Parallel(
        n_jobs=n_jobs,
        verbose=10,
        pre_dispatch="2*n_jobs",
    ) as parallel:
        for start in range(0, len(name_list), batch_size):
            stop = min(start + batch_size, len(name_list))
            results = parallel(
                joblib.delayed(_job)(index)
                for index in range(start, stop)
            )
            preds, gts = zip(*results)
            hist, _ = evaluate.scores(
                gts,
                preds,
                hist,
                num_classes,
            )

    _, score = evaluate.scores([], [], hist, num_classes)
    print("crf score:")
    print(score)
    return True


def main(cfg):
    if args.only_crf:
        return crf_proc(config=cfg)

    val_dataset = coco.CocoSegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage="val",
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers=cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device="cuda",
    )

    trained_state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    (
        gts,
        preds,
        msc_preds,
        cams,
        preds_hist,
        msc_preds_hist,
        cams_hist,
    ) = validate(
        model=model,
        dataset=val_dataset,
        cfg=cfg,
        test_scales=parse_scales(args.scales),
    )
    torch.cuda.empty_cache()

    num_classes = int(cfg.dataset.num_classes)
    preds_hist, seg_score = evaluate.scores(
        gts,
        preds,
        preds_hist,
        num_classes,
    )
    msc_preds_hist, msc_seg_score = evaluate.scores(
        gts,
        msc_preds,
        msc_preds_hist,
        num_classes,
    )
    cams_hist, cam_score = evaluate.scores(
        gts,
        cams,
        cams_hist,
        num_classes,
    )

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    if args.crf:
        crf_proc(config=cfg)

    return True


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    print(cfg)
    print(args)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)
    os.makedirs(os.path.join(args.work_dir, "logit"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "prediction"), exist_ok=True)
    os.makedirs(os.path.join(args.work_dir, "prediction_cmap"), exist_ok=True)

    main(cfg=cfg)
