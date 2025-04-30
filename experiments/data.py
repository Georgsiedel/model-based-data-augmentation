import random
import os
import time
import json

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import (
    Subset,
    Dataset,
    ConcatDataset,
    RandomSampler,
    BatchSampler,
    Sampler,
    DataLoader,
)
import numpy as np
import style_transfer
import experiments.custom_transforms as custom_transforms
from run_exp import device
import gc
from experiments.utils import plot_images, CsvHandler
from experiments.custom_datasets import (
    SubsetWithTransform,
    GeneratedDataset,
    AugmentedDataset,
    ListDataset,
    CustomDataset,
)
from experiments.custom_datasets import (
    BalancedRatioSampler,
    GroupedAugmentedDataset,
    ReproducibleBalancedRatioSampler,
    StyleDataset,
)


def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1):
    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == "CIFAR10":
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == "CIFAR100":
            mean = (
                torch.tensor([0.50707516, 0.48654887, 0.44091784])
                .view(1, 3, 1, 1)
                .to(device)
            )
            std = (
                torch.tensor([0.26733429, 0.25643846, 0.27615047])
                .view(1, 3, 1, 1)
                .to(device)
            )
        elif dataset == "ImageNet" or dataset == "TinyImageNet":
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            print("no normalization values set for this dataset")
    else:
        mean = 0
        std = 1

    return mean, std


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SwaLoader:
    def __init__(self, trainloader, batchsize, robust_samples):
        self.trainloader = trainloader
        self.batchsize = batchsize
        self.robust_samples = robust_samples

    def concatenate_collate_fn(self, batch):
        concatenated_batch = []
        for images, label in batch:
            concatenated_batch.extend(images)
        return torch.stack(concatenated_batch)

    def get_swa_dataloader(self):
        # Create a new DataLoader with the custom collate function

        swa_dataloader = DataLoader(
            dataset=self.trainloader.dataset,
            batch_size=self.batchsize,
            num_workers=0,
            collate_fn=self.concatenate_collate_fn,
            worker_init_fn=self.trainloader.worker_init_fn,
            generator=self.trainloader.generator,
        )

        return swa_dataloader


class DataLoading:
    def __init__(
        self,
        dataset,
        validontest=False,
        epochs=200,
        generated_ratio=0.0,
        resize=False,
        run=0,
        test_only=False,
        factor=1,
        number_workers=0,
        kaggle=False,
    ):
        self.dataset = dataset
        self.generated_ratio = generated_ratio
        self.resize = resize
        self.run = run
        self.epochs = epochs
        self.test_only = test_only
        self.factor = factor
        self.number_workers = number_workers
        self.kaggle = kaggle

        if self.kaggle:
            file_path = os.path.join(os.path.dirname(__file__), "kaggle_path.json")
            with open(file_path, "r") as f:
                self.path = json.load(f)
                self.gen_path = self.path.get(f"{self.dataset}-gen")
                self.corrupt_c_path = self.path.get(f"{self.dataset}-C")
                self.corrupt_bar_path = self.path.get(f"{self.dataset}-C-bar")

    def create_transforms(
        self, train_aug_strat_orig, train_aug_strat_gen, RandomEraseProbability=0.0
    ):
        # list of all data transformations used
        t = transforms.ToTensor()
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        c224 = transforms.RandomCrop(224, padding=28)
        flip = transforms.RandomHorizontalFlip()
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        cc224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(
            p=RandomEraseProbability, scale=(0.02, 0.4)
        )  # , value='random' --> normally distributed and out of bounds 0-1

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset == "ImageNet":
            self.transforms_preprocess = transforms.Compose([t, r256, cc224])
        elif self.resize == True:
            self.transforms_preprocess = transforms.Compose([t, r224])
        else:
            self.transforms_preprocess = transforms.Compose([t])

        # standard augmentations of training set, without tensor transformation
        if self.dataset == "ImageNet":
            self.transforms_basic = transforms.Compose([flip])
        elif self.resize:
            self.transforms_basic = transforms.Compose([flip, c224])
        elif self.dataset == "CIFAR10" or self.dataset == "CIFAR100":
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset == "TinyImageNet":
            self.transforms_basic = transforms.Compose([flip, c64])

        (
            self.stylization_orig,
            self.transforms_orig_after_style,
            self.transforms_orig_after_nostyle,
        ) = custom_transforms.get_transforms_map(
            train_aug_strat_orig, re, self.dataset, self.factor
        )
        (
            self.stylization_gen,
            self.transforms_gen_after_style,
            self.transforms_gen_after_nostyle,
        ) = custom_transforms.get_transforms_map(
            train_aug_strat_gen, re, self.dataset, self.factor
        )

    def load_style_dataloader(self, style_dir, batch_size):
        style_dataset = StyleDataset(style_dir, dataset_type=self.dataset)
        style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=False)
        return style_loader

    def load_base_data(self, validontest, run=0):
        # Trainset and Validset
        if self.test_only == False:
            if self.dataset == "ImageNet" or self.dataset == "TinyImageNet":
                if self.kaggle:
                    self.base_trainset = torchvision.datasets.ImageFolder(
                        root=f"/kaggle/input/tinyimagenet/{self.dataset}/train"
                    )  # Only for TinyImageNet. For use in Kaggle
                else:
                    self.base_trainset = torchvision.datasets.ImageFolder(
                        root=os.path.abspath(f"../data/{self.dataset}/train")
                    )
            else:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.base_trainset = load_helper(
                    root=os.path.abspath("../data"), train=True, download=True
                )

            if validontest == False:
                validsplit = 0.2
                train_indices, val_indices, _, _ = train_test_split(
                    range(len(self.base_trainset)),
                    self.base_trainset.targets,
                    stratify=self.base_trainset.targets,
                    test_size=validsplit,
                    random_state=run,
                )  # same validation split for same runs, but new validation on multiple runs
                self.base_trainset = Subset(self.base_trainset, train_indices)
                self.validset = Subset(self.base_trainset, val_indices)
                self.validset = list(map(self.transforms_preprocess, self.validset))
            else:
                if self.dataset == "ImageNet" or self.dataset == "TinyImageNet":
                    self.validset = torchvision.datasets.ImageFolder(
                        root=f"/kaggle/input/tinyimagenet/{self.dataset}/val",
                        transform=self.transforms_preprocess,
                    )
                elif self.dataset == "CIFAR10" or self.dataset == "CIFAR100":
                    load_helper = getattr(torchvision.datasets, self.dataset)
                    self.validset = load_helper(
                        root=os.path.abspath("../data"),
                        train=False,
                        download=True,
                        transform=self.transforms_preprocess,
                    )
                else:
                    print("Dataset not loadable")
        else:
            self.trainset = None
            self.validset = None

        # Testset
        if self.dataset == "ImageNet" or self.dataset == "TinyImageNet":
            self.testset = torchvision.datasets.ImageFolder(
                root=f"/kaggle/input/tinyimagenet/{self.dataset}/val",
                transform=self.transforms_preprocess,
            )
        elif self.dataset == "CIFAR10" or self.dataset == "CIFAR100":
            load_helper = getattr(torchvision.datasets, self.dataset)
            self.testset = load_helper(
                root=os.path.abspath("../data"),
                train=False,
                download=True,
                transform=self.transforms_preprocess,
            )
        else:
            print("Dataset not loadable")

        self.num_classes = len(self.testset.classes)

    def load_augmented_traindata(self, target_size, epoch=0, robust_samples=0):
        self.robust_samples = robust_samples
        self.target_size = target_size
        if self.kaggle:
            self.generated_dataset = np.load(self.gen_path, mmap_mode="r")
        else:
            self.generated_dataset = (
                np.load(
                    os.path.abspath(f"../data/{self.dataset}-add-1m-dm.npz"),
                    mmap_mode="r",
                )
                if self.generated_ratio > 0.0
                else None
            )
        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        torch.cuda.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        self.num_generated = int(target_size * self.generated_ratio)
        self.num_original = target_size - self.num_generated

        if self.num_original > 0:
            original_indices = torch.randperm(self.target_size)[: self.num_original]
            original_subset = SubsetWithTransform(
                Subset(self.base_trainset, original_indices), self.transforms_preprocess
            )

            if self.stylization_orig is not None:
                stylized_original_subset, style_mask_orig = self.stylization_orig(
                    original_subset
                )
            else:
                stylized_original_subset, style_mask_orig = (
                    original_subset,
                    [False] * len(original_subset),
                )
        else:
            stylized_original_subset, style_mask_orig = None, []

        if self.num_generated > 0 and self.generated_dataset is not None:
            generated_indices = np.random.choice(
                len(self.generated_dataset["label"]),
                size=self.num_generated,
                replace=False,
            )

            generated_subset = GeneratedDataset(
                self.generated_dataset["image"][generated_indices],
                self.generated_dataset["label"][generated_indices],
                transform=self.transforms_preprocess,
            )

            if self.stylization_gen is not None:
                stylized_generated_subset, style_mask_gen = self.stylization_gen(
                    generated_subset
                )
            else:
                stylized_generated_subset, style_mask_gen = (
                    generated_subset,
                    [False] * len(generated_subset),
                )
        else:
            stylized_generated_subset, style_mask_gen = None, []

        style_mask = style_mask_orig + style_mask_gen

        self.trainset = AugmentedDataset(
            stylized_original_subset,
            stylized_generated_subset,
            style_mask,
            self.transforms_basic,
            self.transforms_orig_after_style,
            self.transforms_gen_after_style,
            self.transforms_orig_after_nostyle,
            self.transforms_gen_after_nostyle,
            self.robust_samples,
        )

    def load_data_c(self, subset, subsetsize, valid_run):
        c_datasets = []
        current_dir = os.path.dirname(__file__)

        # Load corruption labels
        c_path = os.path.join(current_dir, "../data/c-labels.txt")
        corruptions_c = np.asarray(np.loadtxt(c_path, dtype=list))

        np.random.seed(self.run)
        torch.manual_seed(self.run)
        random.seed(self.run)
        global fixed_worker_rng
        fixed_worker_rng = np.random.default_rng()

        if self.dataset in ["CIFAR10", "CIFAR100"]:
            # Load c-bar corruptions
            c_bar_path = os.path.join(current_dir, "../data/c-bar-labels-cifar.txt")
            csv_handler = CsvHandler(
                os.path.join(current_dir, "../data/cifar_c_bar.csv")
            )
            corruptions_bar = csv_handler.read_corruptions()
            corruptions = [(s, "c") for s in corruptions_c] + [
                (s, "c-bar") for s in corruptions_bar
            ]

            for corruption, set_name in corruptions:
                if self.validontest:
                    subtestset = self.testset

                    # Load .npy corruption file with Kaggle-aware paths
                    try:
                        if self.kaggle:
                            try:
                                np_data_c = np.load(
                                    os.path.join(
                                        self.corrupt_c_path, f"{corruption}.npy"
                                    )
                                )
                            except FileNotFoundError:
                                np_data_c = np.load(
                                    os.path.join(
                                        self.corrupt_bar_path, f"{corruption}.npy"
                                    )
                                )
                        else:
                            np_data_c = np.load(
                                os.path.abspath(
                                    f"../data/{self.dataset}-{set_name}/{corruption}.npy"
                                )
                            )
                    except FileNotFoundError as e:
                        raise FileNotFoundError(
                            f"Missing corruption file: {corruption}"
                        ) from e

                    np_data_c = np.array(np.array_split(np_data_c, 5))

                    if subset:
                        selected_indices = np.random.choice(
                            len(self.testset), subsetsize, replace=False
                        )
                        subtestset = Subset(self.testset, selected_indices)
                        np_data_c = [
                            intensity_dataset[selected_indices]
                            for intensity_dataset in np_data_c
                        ]

                    concat_intensities = ConcatDataset(
                        [
                            CustomDataset(
                                intensity_data_c,
                                subtestset,
                                self.resize,
                                self.transforms_preprocess,
                            )
                            for intensity_data_c in np_data_c
                        ]
                    )
                    c_datasets.append(concat_intensities)

                else:
                    # Random corruption applied on-the-fly
                    corrupted_set = SubsetWithTransform(
                        self.testset,
                        transform=custom_transforms.RandomCommonCorruptionTransform(
                            set_name, corruption, self.dataset, csv_handler
                        ),
                    )

                    if subset:
                        selected_indices = np.random.choice(
                            len(self.testset), subsetsize, replace=False
                        )
                        corrupted_set = Subset(corrupted_set, selected_indices)

                    if valid_run:
                        # Cache transformed samples
                        r = torch.Generator()
                        r.manual_seed(0)
                        precompute_loader = DataLoader(
                            corrupted_set,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=self.number_workers,
                            worker_init_fn=seed_worker,
                            generator=r,
                        )

                        if corruption in ["caustic_refraction", "sparkles"]:
                            precomputed_samples = [
                                (sample[0], label[0])
                                for sample, label in precompute_loader
                            ]
                        else:
                            precomputed_samples = [sample for sample in corrupted_set]

                        corrupted_set = ListDataset(precomputed_samples)

                    c_datasets.append(corrupted_set)

        else:
            print("No corrupted benchmark available other than CIFAR10-c, CIFAR100-c.")
            return

        if subset:
            c_datasets = ConcatDataset(c_datasets)
            self.c_datasets_dict = {"combined": c_datasets}
        else:
            self.c_datasets_dict = {
                label: dataset
                for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)
            }

        return self.c_datasets_dict

    def get_loader(self, batchsize):
        self.batchsize = batchsize

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        print(
            f"[GET_LOADER] Batchsize: {batchsize}, Generated ratio: {self.generated_ratio}, Epoch: {self.epoch}, Run: {self.run}"
        )

        if self.generated_ratio > 0.0:
            self.CustomSampler = BalancedRatioSampler(
                self.trainset,
                generated_ratio=self.generated_ratio,
                batch_size=batchsize,
            )
        else:
            self.CustomSampler = BatchSampler(
                RandomSampler(self.trainset), batch_size=batchsize, drop_last=False
            )

        print(
            f"Batchsize: {batchsize}, Generated ratio: {self.generated_ratio}, Epoch: {self.epoch}, Run: {self.run}"
        )

        self.trainloader = DataLoader(
            self.trainset,
            pin_memory=True,
            batch_sampler=self.CustomSampler,
            num_workers=self.number_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
        )

        val_workers = self.number_workers if self.dataset == "ImageNet" else 0
        self.testloader = DataLoader(
            self.testset, batch_size=batchsize, pin_memory=True, num_workers=val_workers
        )

        return self.trainloader, self.testloader

    def update_set(self, epoch, start_epoch):
        if (
            (
                self.generated_ratio != 0.0
                or self.stylization_gen is not None
                or self.stylization_orig is not None
            )
            and epoch != 0
            and epoch != start_epoch
        ):
            del self.trainset

            self.load_augmented_traindata(
                self.target_size, epoch=epoch, robust_samples=self.robust_samples
            )

        del self.trainloader
        gc.collect()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(
            self.trainset,
            batch_sampler=self.CustomSampler,
            pin_memory=True,
            num_workers=self.number_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
        )

        return self.trainloader

    def update_set_grouped(self, epoch, start_epoch):
        if (self.generated_ratio != 0.0) and epoch != 0 and epoch != start_epoch:
            self.load_augmented_traindata(
                self.target_size, epoch=epoch, robust_samples=self.robust_samples
            )
        elif (
            (self.stylization_gen is not None or self.stylization_orig is not None)
            and epoch != 0
            and epoch != start_epoch
        ):
            self.trainset.set_epoch(epoch)

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(
            self.trainset,
            batch_sampler=self.CustomSampler,
            pin_memory=True,
            num_workers=self.number_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
        )
        return self.trainloader

    def get_loader_grouped(self, batchsize):
        self.batchsize = batchsize

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        self.CustomSampler = ReproducibleBalancedRatioSampler(
            self.trainset,
            generated_ratio=self.generated_ratio,
            batch_size=batchsize,
            epoch=self.epoch,
        )

        self.trainloader = DataLoader(
            self.trainset,
            pin_memory=True,
            batch_sampler=self.CustomSampler,
            num_workers=self.number_workers,
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=False,
        )

        val_workers = self.number_workers if self.dataset == "ImageNet" else 0
        self.validationloader = DataLoader(
            self.validset,
            batch_size=batchsize,
            pin_memory=True,
            num_workers=val_workers,
        )

        return self.trainloader, self.validationloader

    def load_augmented_traindata_grouped(self, target_size, epoch=0, robust_samples=0):
        self.robust_samples = robust_samples
        self.target_size = target_size
        self.generated_dataset = (
            np.load(
                os.path.abspath(f"../data/{self.dataset}-add-1m-dm.npz"), mmap_mode="r"
            )
            if self.generated_ratio > 0.0
            else None
        )
        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        self.num_generated = int(target_size * self.generated_ratio)
        self.num_original = target_size - self.num_generated

        if self.num_original > 0:
            original_indices = torch.randperm(len(self.base_trainset))[
                : self.num_original
            ]
            original_subset = SubsetWithTransform(
                Subset(self.base_trainset, original_indices), self.transforms_preprocess
            )
        else:
            original_subset = None

        if self.num_generated > 0 and self.generated_dataset is not None:
            generated_indices = np.random.choice(
                len(self.generated_dataset["label"]),
                size=self.num_generated,
                replace=False,
            )

            generated_subset = GeneratedDataset(
                self.generated_dataset["image"][generated_indices],
                self.generated_dataset["label"][generated_indices],
                transform=self.transforms_preprocess,
            )

        else:
            generated_subset = None

        self.trainset = GroupedAugmentedDataset(
            original_subset,
            generated_subset,
            self.transforms_basic,
            self.stylization_orig,
            self.stylization_gen,
            self.transforms_orig_after_style,
            self.transforms_gen_after_style,
            self.transforms_orig_after_nostyle,
            self.transforms_gen_after_nostyle,
            self.robust_samples,
            epoch,
        )
