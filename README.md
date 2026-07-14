# CRST: Cross-Resolution Semantic Transfer for Robust Text-to-Image Person Retrieval

This repository contains the PyTorch implementation of **Cross-Resolution Semantic Transfer (CRST)** for robust text-to-image person retrieval under low-resolution and mixed-resolution surveillance scenarios.

CRST is built on a CLIP-style dual encoder and addresses two cross-resolution failure modes: **Evidence Reliability Collapse (ERC)**, where low-resolution images corrupt fine-grained visual evidence for language grounding, and **Ranking Distribution Drift (RDD)**, where mixed HR/LR galleries reshape similarity neighborhoods and destabilize retrieval rankings.

## Framework

<p align="center">
  <img src="figs/overall-framework.png" width="900">
</p>

CRST contains three training-time components:

- **Resolution-Conditioned Reasoner (RCR):** estimates token-wise visual reliability for reliability-aware masked grounding.
- **Text-Guided Feature Refiner (TGR):** uses text tokens as semantic priors to correct degraded LR image embeddings.
- **Cross-Resolution Ranking Distribution Alignment (CR-RDA):** transfers HR ranking geometry to LR retrieval by aligning ranking distributions.

At inference, CRST keeps the standard dual-encoder retrieval form: image and text embeddings are extracted independently and ranked by cosine similarity. Therefore, CRST does not introduce query-dependent interaction or extra inference-stage restoration cost.

## Requirements

```bash
conda create -n crst python=3.8 -y
conda activate crst
pip install -r requirements.txt
```

The CLIP BPE vocabulary file is required by `utils/simple_tokenizer.py`. If `data/bpe_simple_vocab_16e6.txt.gz` is missing, the tokenizer will try to download it automatically. In offline environments, place it manually at:

```text
data/bpe_simple_vocab_16e6.txt.gz
```

## Training

### CUHK-PEDES

```bash
python train.py --name crst_cuhk --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id' --dataset_name 'CUHK-PEDES' --root_dir '/path/to/datasets' --num_epoch 60 --paired_loss_weight 1.0 --feat_loss_weight 0.1 --cr_rda_loss_weight 0.1
```

### ICFG-PEDES

```bash
python train.py --name crst_icfg --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id' --dataset_name 'ICFG-PEDES' --root_dir '/path/to/datasets' --num_epoch 60 --paired_loss_weight 1.0 --feat_loss_weight 0.1 --cr_rda_loss_weight 0.1
```

During training, the HR branch is used as the high-fidelity reference, while the paired branch randomly samples HR/Mild-LR/Mid-LR/Ultra-LR views.

## Evaluation

### HR

```bash
python test.py --config_file "./logs/CUHK-PEDES/your_trained_run/configs.yaml" --test_res_label 0
```

### Ultra-LR

```bash
python test.py --config_file "./logs/CUHK-PEDES/your_trained_run/configs.yaml" --test_img_root /root/datasets/CUHK-PEDES/imgs_UltraLR --test_res_label 3
```

### Mixed

```bash
python test.py --config_file "./logs/CUHK-PEDES/your_trained_run/configs.yaml" --test_img_root /root/datasets/CUHK-PEDES/imgs_Multi --test_res_label 4
```

For `--test_res_label 4`, the code loads the corresponding `imgs_Multi_mapping.txt` file and assigns each gallery image its own resolution label.

## Citing CRST

```bibtex
@article{qian2026cross,
  title={Cross-Resolution Semantic Transfer for Robust Text-to-Image Retrieval in Low-Resolution Surveillance},
  author={Qian, Wenjie and Yang, Bin and Wang, Xiao and Huang, Wenke and Mei, Ling and Xu, Xin and Ye, Mang},
  journal={arXiv preprint arXiv:2606.30458},
  year={2026}
}
```
