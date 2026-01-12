This is the repository for the paper "Stylized Synthetic Augmentation further improves Corruption Robustness", available here: https://arxiv.org/abs/2512.15675
We train image classification models with additional synthetic data as well as stylization. The repository allows to configure a multitude of additional data augmentation strategies.

run_exp.py calls train.py and eval.py modules from the experiments folder for one or multiple experiment IDs, the setup of which need to be defined in experiments/configs/config_{ID}.py

paths.json lets you adjusts the paths to important directories containing data in order to allow other repository structures or the use of e.g. Kaggle.
As standard, the paths defined reference "data" and "trained_models" repositories on the same directory level as the project (one level up).

CIFAR10 / CIFAR100 are downloaded automatically to "data", but all ImageNet, TinyImageNet, -c and -c-bar datasets need to be added.
Similarly, generated data usage (setting generate_ratio >0.0) requires the respective images in "data" in numpy format: "{dataset}-add-1m-dm.npz" as can be downloaded from here: https://github.com/wzekai99/DM-Improves-AT or generated here: https://github.com/NVlabs/edm

The folder "data" given here contains information for the c- and c-bar datasets, which must also be in the final "data" repository with all other data as described above. Note that currently, the "data"-path according to paths.json is one level above!

Stylization in our implementation requires encoded image features from the painter-by-numbers dataset to be put into the data repository and named "style_feats_adain_1000.npy". For reproduction of our results, download the 1000 image features used by us from here: https://zenodo.org/records/16279015

The model architectures in /experiments/models contain a parameter "factor" for TinyImageNet's 64x64 images. This model uses the same architecture as for CIFAR 32x32 images, just with a stride=factor=2 in the first convolution. All models inherit a forward pass from ct_model.py to allow normalization, noise injections and mixup within the forward pass (and in deeper layers).

