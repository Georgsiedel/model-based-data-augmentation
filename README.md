# Stylized Synthetic Augmentation Further Improves Corruption Robustness

This repository accompanies the paper **“Stylized Synthetic Augmentation Further Improves Corruption Robustness”**, available here:  
https://arxiv.org/abs/2512.15675

We train image classification models using additional synthetic data and stylization. The repository provides flexible configuration of multiple data augmentation strategies and experiment setups.

---

## 🧭 Overview

- Training of corruption-robust image classifiers  
- Support for **synthetic data augmentation**  
- Support for **stylization-based augmentation**  
- Flexible experiment configuration  
- Works with multiple datasets and repository structures

---

## 📂 Repository Structure

- `run_exp.py` – main experiment launcher  
- `experiments/`
  - `train.py` – training script  
  - `eval.py` – evaluation script  
  - `configs/config_{ID}.py` – experiment configuration files  
- `experiments/models/` – model definitions  
- `paths.json` – configuration for dataset and checkpoint paths  
- `data/` – contains information for c and c-bar datasets

---

## ▶️ Running Experiments

`run_exp.py` runs one or multiple experiment IDs.

Each experiment setup must be defined in
`experiments/configs/config_{ID}.py`


Internally, the launcher calls:

- `experiments/train.py`
- `experiments/eval.py`

---

## 🛠 Path Configuration

Use `paths.json` to specify directories for:

- datasets
- pretrained or trained models
- external storage layouts (e.g., Kaggle, custom structures)

Default expectation:

project_root/

├── repository/

├── data/

└── trained_models/


> The `data/` folder inside this repository only contains information for c and c-bar datasets; full datasets must be placed in the external `data/` directory referenced in `paths.json`.

---

## 📚 Datasets

### Automatically downloaded
- CIFAR-10  
- CIFAR-100  

Both are placed automatically into `data/`.

### Must be added manually
- ImageNet  
- TinyImageNet  
- Corrupted variants:
  - `-c`
  - `-c-bar`

---

## 🧪 Synthetic Data Usage

To enable generated data (`generate_ratio > 0.0`), place `.npz` files in `data/` with the naming pattern:
`{dataset}-add-1m-dm.npz`


They can be obtained from:

- https://github.com/wzekai99/DM-Improves-AT  
or generated via:

- https://github.com/NVlabs/edm

---

## 🎨 Stylization Features

Stylization requires encoded image features from **Painter-by-Numbers**.

Required file in `data/`:
`style_feats_adain_1000.npy`


For exact reproduction, download the 1000 features used here:

- https://zenodo.org/records/16279015

---

## 🧭 Model Architectures

Models are located in:
`experiments/models/`


Key characteristics:

- include parameter `factor` for TinyImageNet (64×64)
- same base architecture as CIFAR (32×32)
- first convolution uses stride = `factor = 2` for TinyImageNet
- all models inherit forward pass from `ct_model.py`, enabling:
  - normalization  
  - noise injection  
  - mixup  
  - deeper-layer augmentations  

---

## ✅ Capabilities Summary

- corruption-robust training  
- integration of synthetic and stylized data  
- configurable experiment setups  
- flexible path handling  
- unified augmentation control inside the forward pass

---


