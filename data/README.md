Example datafolder, current path in repo leads outside the workspace.
CIFAR10 / CIFAR100 are downloaded automatically, all ImageNet, TinyImageNet, -c and -c-bar datasets need to be added.

Generated data usage requires the respective images in this folder in numpy format: "{dataset}-add-1m-dm.npz" 
as can be downloaded from here: https://github.com/wzekai99/DM-Improves-AT 
or generated here: https://github.com/NVlabs/edm

Stylemix in our implementation uses the already encoded features of a selection of 1000 paintings images: "style_feats_adain_1000.npy"
as can be downloaded from here: tba