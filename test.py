from prettytable import PrettyTable
import os
import torch
import os.path as op
from pathlib import Path

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs

RES_MAP = {
    "imgs": 0,
    "imgs_MildLR": 1,
    "imgs_MidLR": 2,
    "imgs_UltraLR": 3,
}


def _infer_original_img_root(dataset):
    """Return the original image directory used when the test dataset was built."""
    if hasattr(dataset, "img_paths") and len(dataset.img_paths) > 0:
        # Dataset-specific roots end with args.img_subdir, normally "imgs".
        # We keep the part before the first relative test path by using the
        # common directory over all absolute image paths.
        return Path(op.commonpath(dataset.img_paths))
    raise RuntimeError("Can not infer image root from an empty image dataset.")


def _replace_test_image_root(test_img_loader, new_root):
    dataset = test_img_loader.dataset
    old_paths = [Path(p) for p in dataset.img_paths]
    new_root = Path(new_root).expanduser().resolve()

    if not new_root.exists():
        raise FileNotFoundError(f"test_img_root does not exist: {new_root}")

    # In CUHK/ICFG/RSTP paths, the image root is the directory named by
    # args.img_subdir, usually "imgs". Replace that path component only,
    # rather than using string.replace('imgs', ...), which can corrupt paths.
    old_root_name = getattr(dataset, "img_subdir", None) or "imgs"
    new_paths = []
    for p in old_paths:
        parts = list(p.parts)
        if old_root_name in parts:
            idx = len(parts) - 1 - parts[::-1].index(old_root_name)
            rel = Path(*parts[idx + 1:])
            new_paths.append(str(new_root / rel))
        else:
            # Fallback: use the relative path under the common directory.
            common_root = _infer_original_img_root(dataset)
            rel = p.relative_to(common_root)
            new_paths.append(str(new_root / rel))

    dataset.img_paths = new_paths
    return new_root


def _find_multi_mapping_file(args, test_img_root):
    root_name = test_img_root.name
    dataset_dir = Path(args.root_dir) / args.dataset_name
    candidates = [
        test_img_root / f"{root_name}_mapping.txt",
        test_img_root.parent / f"{root_name}_mapping.txt",
        dataset_dir / f"{root_name}_mapping.txt",
        # Common aliases used in earlier CRST experiments.
        test_img_root.parent / "imgs_Multi_mapping.txt",
        test_img_root.parent / "imgs_Mixed_mapping.txt",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Can not find multi-resolution mapping file. Tried: " + ", ".join(str(p) for p in candidates))


def _load_multi_labels(args, test_img_loader, test_img_root, strict=True):
    mapping_file = _find_multi_mapping_file(args, test_img_root)
    path_to_label = {}
    with open(mapping_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            rel_path, res_name = parts[0], parts[1]
            if res_name in RES_MAP:
                path_to_label[rel_path.replace("\\", "/")] = RES_MAP[res_name]

    labels = []
    missing = []
    for abs_path in test_img_loader.dataset.img_paths:
        try:
            rel = Path(abs_path).resolve().relative_to(test_img_root).as_posix()
        except ValueError:
            rel = Path(abs_path).as_posix()
        if rel not in path_to_label:
            missing.append(rel)
            labels.append(3)
        else:
            labels.append(path_to_label[rel])

    if missing and strict:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{len(missing)} images are missing in {mapping_file}. Examples: {preview}")

    return labels, mapping_file, len(missing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CRST Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/crst/configs.yaml')
    parser.add_argument("--test_img_root", default="", help="path to the test image dir, e.g., data/CUHK-PEDES/imgs_UltraLR")
    parser.add_argument("--test_res_label", type=int, default=0, help="0:HR, 1:Mild, 2:Mid, 3:Ultra, 4:Mixed/Multi")
    parser.add_argument("--allow_missing_multi_mapping", action="store_true", help="default missing mixed mapping entries to Ultra-LR instead of raising an error")
    parser.add_argument("--dump_emb", action="store_true", help="dump embeddings to a .pt file")
    parser.add_argument("--emb_out", default="emb_test.pt", help="output embedding filename; saved under args.output_dir unless absolute")

    args_cli = parser.parse_args()

    args = load_train_configs(args_cli.config_file)
    args.test_img_root = args_cli.test_img_root
    args.test_res_label = args_cli.test_res_label
    args.training = False

    logger = setup_logger('CRST', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    if args.test_res_label == 4 and args.test_img_root == "":
        raise ValueError("Mixed/Multi evaluation requires --test_img_root pointing to imgs_Multi or imgs_Mixed.")

    device = "cuda"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args, test_res_label=args.test_res_label)

    if args.test_img_root != "":
        test_img_root = _replace_test_image_root(test_img_loader, args.test_img_root)
        logger.info(f"Overriding test image root with: {test_img_root}")

        if args.test_res_label == 4:
            logger.info("Detected Mixed/Multi-resolution evaluation. Loading image-wise resolution labels.")
            labels, mapping_file, missing = _load_multi_labels(
                args,
                test_img_loader,
                test_img_root,
                strict=not args_cli.allow_missing_multi_mapping,
            )
            if missing > 0:
                logger.warning(f"{missing} images were missing in the mapping and defaulted to Ultra-LR(3).")
            logger.info(f"Loaded {len(labels)} image-wise labels from {mapping_file}.")
            test_img_loader.dataset.res_label = labels

    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    if args_cli.dump_emb:
        evaluator = Evaluator(test_img_loader, test_txt_loader)
        qfeats, gfeats, qids, gids = evaluator._compute_embedding(model)

        out_path = args_cli.emb_out
        if not os.path.isabs(out_path):
            out_path = op.join(args.output_dir, out_path)

        os.makedirs(op.dirname(out_path) or ".", exist_ok=True)
        torch.save(
            {
                "qfeats": qfeats.detach().cpu(),
                "gfeats": gfeats.detach().cpu(),
                "qids": qids.detach().cpu(),
                "gids": gids.detach().cpu(),
            },
            out_path,
        )
        logger.info(f"[Dump Embeddings] saved to: {out_path}")

    do_inference(model, test_img_loader, test_txt_loader)
