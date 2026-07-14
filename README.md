# CRST: Cross-Resolution Semantic Transfer for Robust Text-to-Image Person Retrieval

This repository provides the PyTorch implementation of **Cross-Resolution Semantic Transfer (CRST)** for robust text-to-image person retrieval under low-resolution and mixed-resolution surveillance settings.

CRST is built on a CLIP-style dual encoder and targets two cross-resolution failure modes:

- **Evidence Reliability Collapse (ERC):** low-resolution images corrupt fine-grained visual evidence needed for language grounding.
- **Ranking Distribution Drift (RDD):** mixed HR/LR galleries reshape the similarity neighborhood and destabilize rankings.

To address them, CRST uses three training-time components:

1. **Resolution-Conditioned Reasoner (RCR):** predicts token-wise visual reliability and performs reliability-aware masked grounding.
2. **Text-Guided Feature Refiner (TGR):** uses text tokens as semantic priors to correct degraded LR image embeddings.
3. **Cross-Resolution Ranking Distribution Alignment (CR-RDA):** transfers HR ranking geometry to LR retrieval by aligning ranking distributions.

At inference, CRST keeps the standard dual-encoder retrieval form: image and text embeddings are extracted independently and ranked by cosine similarity. RCR, TGR, feature consistency, and CR-RDA are training-time constraints and do not introduce query-dependent inference interaction.

## Requirements

The code was developed with PyTorch and a CLIP ViT-B/16 backbone. A typical environment is:

```bash
conda create -n crst python=3.8 -y
conda activate crst
pip install -r requirements.txt
```

The CLIP BPE vocabulary file is required by `utils/simple_tokenizer.py`. If `data/bpe_simple_vocab_16e6.txt.gz` is missing, the tokenizer will try to download it automatically. In offline environments, download it manually and place it at:

```text
data/bpe_simple_vocab_16e6.txt.gz
```

## Dataset Preparation

Organize datasets under one root directory:

```text
<root_dir>/
├── CUHK-PEDES/
│   ├── imgs/
│   ├── imgs_MildLR/
│   ├── imgs_MidLR/
│   ├── imgs_UltraLR/
│   ├── imgs_Multi/
│   ├── imgs_Multi_mapping.txt
│   └── reid_raw.json
├── ICFG-PEDES/
│   ├── imgs/
│   └── ICFG-PEDES.json
└── RSTPReid/
    ├── imgs/
    └── data_captions.json
```

The cross-resolution protocol uses four image levels:

| label | setting | resolution protocol |
|---:|---|---|
| 0 | HR | original image resized to 384×128 |
| 1 | Mild-LR | 128×64 → resized back to 384×128 |
| 2 | Mid-LR | 64×32 → resized back to 384×128 |
| 3 | Ultra-LR | 32×16 → resized back to 384×128 |
| 4 | Mixed/Multi | each gallery image uses its own label from the mapping file |

For Mixed/Multi evaluation, the mapping file should be tab-separated:

```text
# rel_path    chosen_res_dir
cam_a/000_45.bmp    imgs
cam_a/002_45.bmp    imgs_MidLR
cam_a/008_45.bmp    imgs_UltraLR
```

## Training

### CUHK-PEDES

```bash
python train.py \
  --name crst_cuhk \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --loss_names 'sdm+mlm+id' \
  --dataset_name 'CUHK-PEDES' \
  --root_dir '/path/to/datasets' \
  --num_epoch 60 \
  --paired_loss_weight 1.0 \
  --feat_loss_weight 0.1 \
  --cr_rda_loss_weight 0.1
```

### ICFG-PEDES

```bash
python train.py \
  --name crst_icfg \
  --img_aug \
  --batch_size 64 \
  --MLM \
  --loss_names 'sdm+mlm+id' \
  --dataset_name 'ICFG-PEDES' \
  --root_dir '/path/to/datasets' \
  --num_epoch 60 \
  --paired_loss_weight 1.0 \
  --feat_loss_weight 0.1 \
  --cr_rda_loss_weight 0.1
```

During training, the HR branch always uses the HR image as a stop-gradient reference, while the paired LR branch randomly samples HR/Mild/Mid/Ultra views.

## Evaluation

Set the trained log directory first:

```bash
LOG=logs/CUHK-PEDES/your_trained_run
```

### HR

```bash
python test.py \
  --config_file "$LOG/configs.yaml" \
  --test_res_label 0
```

### Mild-LR

```bash
python test.py \
  --config_file "$LOG/configs.yaml" \
  --test_img_root /path/to/datasets/CUHK-PEDES/imgs_MildLR \
  --test_res_label 1
```

### Mid-LR

```bash
python test.py \
  --config_file "$LOG/configs.yaml" \
  --test_img_root /path/to/datasets/CUHK-PEDES/imgs_MidLR \
  --test_res_label 2
```

### Ultra-LR

```bash
python test.py \
  --config_file "$LOG/configs.yaml" \
  --test_img_root /path/to/datasets/CUHK-PEDES/imgs_UltraLR \
  --test_res_label 3
```

### Mixed / Multi

```bash
python test.py \
  --config_file "$LOG/configs.yaml" \
  --test_img_root /path/to/datasets/CUHK-PEDES/imgs_Multi \
  --test_res_label 4
```

For `--test_res_label 4`, the code loads the corresponding `imgs_Multi_mapping.txt` file and assigns each gallery image its own resolution label.

## Ablation

You can disable individual CRST components:

```bash
--no_rcr       # disable RCR and use standard masked grounding
--no_tgr       # disable Text-Guided Feature Refiner
--no_feat      # disable HR-referenced feature consistency
--no_cr_rda    # disable Cross-Resolution Ranking Distribution Alignment
--no_res_embed # disable visual resolution embedding
```

Backward-compatible aliases are also kept:

```bash
--no_refiner   # alias of --no_tgr
--no_distill   # disables feature consistency and CR-RDA
```

## Repository Structure

```text
model/build.py          # CRST model wrapper and training forward pass
model/crst_modules.py   # RCR and TGR modules
model/objectives.py     # SDM, MLM, ID, feature consistency, CR-RDA losses
datasets/bases.py       # paired HR/LR datasets and resolution-conditioned masking
datasets/preprocessing.py # random HR/Mild/Mid/Ultra view generation
test.py                 # HR/LR/Mixed evaluation entry
train.py                # training entry
```

## Citation

If you use this code, please cite the CRST paper.

```bibtex
@article{qian2026cross,
  title={Cross-Resolution Semantic Transfer for Robust Text-to-Image Retrieval in Low-Resolution Surveillance},
  author={Qian, Wenjie and Yang, Bin and Wang, Xiao and Huang, Wenke and Mei, Ling and Xu, Xin and Ye, Mang},
  journal={arXiv preprint arXiv:2606.30458},
  year={2026}
}
```
