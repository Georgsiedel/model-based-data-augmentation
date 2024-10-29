run_exp.py is the test execution that calls train.py and eval.py modules from the experiments folder and allows to execute multiple experiments and runs of the same experiments successively.

run_exp.py uses parameters for every experiment that are defined in a config file, stored at experiments/configs. Every config file has a number, e.g. config0.py which is called as the "experiment parameter in run_exp.py.

a data folder on the same level as the main repository is required, containing all the datasets for training and evaluation.

We use a sub-folder structure for trained models and their results: /results/['datasetname']/['modelname'] and /experiments/trained_models/['datasetname']/['modelname']. Please notice the Readme in said folders, as empty folders are not allowed. 

The model architectures in /experiments/models are reworked with parameters such as a size factor for TinyImageNet 64x64 images using the same architecture as for CIFAR 32x32 images, as well as SiLu functions. Models also inherit a forward pass from ct_model.py to allow normalization at the model level, noise injections and mixup within the forward pass (and in deeper layers). Some model architectures may need to be adjusted to inherit from ct_model.py.
