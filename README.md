# <div align="center">CRST: Cross-Resolution Semantic Transfer<br>for Robust Text-to-Image Person Retrieval</div>

<div align="center">

🔥🔥 CRST is accepted by ACM MM 2026! 🔥🔥

<a href="https://arxiv.org/pdf/2606.30458">
  <img src="https://img.shields.io/badge/arXiv-2606.30458-b31b1b.svg" alt="arXiv">
</a>

</div>

> **Important Note:** This repository is the official PyTorch implementation of **Cross-Resolution Semantic Transfer (CRST)**.  
> CRST is designed for robust text-to-image person retrieval under low-resolution and mixed-resolution surveillance scenarios.

---

## Introduction

🌟 **Cross-resolution text-to-image person retrieval**

CRST studies a practical surveillance setting where high-resolution and low-resolution pedestrian images coexist in the same gallery. This setting introduces two key challenges:

- **Evidence Reliability Collapse:** low-resolution images corrupt fine-grained visual evidence required by text descriptions.
- **Ranking Distribution Drift:** mixed-resolution galleries reshape the retrieval neighborhood and destabilize ranking orders.

To address these challenges, CRST transfers high-resolution semantic evidence to low-resolution retrieval through reliability-aware reasoning, text-guided feature correction, and ranking-distribution alignment.

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
python train.py --name crst_cuhk --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id' --dataset_name 'CUHK-PEDES' --root_dir '/root/datasets' --num_epoch 60 --paired_loss_weight 1.0 --feat_loss_weight 0.1 --cr_rda_loss_weight 0.1
```

### ICFG-PEDES

```bash
python train.py --name crst_icfg --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id' --dataset_name 'ICFG-PEDES' --root_dir '/root/datasets' --num_epoch 60 --paired_loss_weight 1.0 --feat_loss_weight 0.1 --cr_rda_loss_weight 0.1
```

### RSTPReid

```bash
python train.py --name crst_rstp --img_aug --batch_size 64 --MLM --loss_names 'sdm+mlm+id' --dataset_name 'RSTPReid' --root_dir '/root/datasets' --num_epoch 60 --paired_loss_weight 1.0 --feat_loss_weight 0.1 --cr_rda_loss_weight 0.1
```

## Evaluation

### HR

```bash
python test.py --config_file "./logs/CUHK-PEDES/trained_run/configs.yaml" --test_res_label 0
```

### Ultra-LR

```bash
python test.py --config_file "./logs/CUHK-PEDES/trained_run/configs.yaml" --test_img_root /root/datasets/CUHK-PEDES/imgs_UltraLR --test_res_label 3
```

### Mixed

```bash
python test.py --config_file "./logs/CUHK-PEDES/trained_run/configs.yaml" --test_img_root /root/datasets/CUHK-PEDES/imgs_Multi --test_res_label 4
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
