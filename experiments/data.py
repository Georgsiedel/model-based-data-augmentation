import random
import os
import time

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset, RandomSampler, BatchSampler, Sampler, DataLoader
import numpy as np
import style_transfer
import experiments.custom_transforms as custom_transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1):

    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == 'CIFAR100':
            mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1).to(device)
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet'):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            print('no normalization values set for this dataset')
    else:
        mean = 0
        std = 1

    return mean, std

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class SwaLoader():
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
            generator=self.trainloader.generator
        )

        return swa_dataloader

class CustomDataset(Dataset):
    def __init__(self, np_images, original_dataset, resize):
        # Load images
        self.images = torch.from_numpy(np_images).permute(0, 3, 1, 2) / 255
        if resize == True:
            self.images = transforms.Resize(224, antialias=True)(self.images)

        # Extract labels from the original PyTorch dataset
        self.labels = [label for _, label in original_dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.images[index]
        label = self.labels[index]

        return image, label

class BalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size, group_size=32):
        super(BalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.group_size = group_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch

    def _create_groups(self, num_items, offset=0):
        """Create reproducible groups by dividing the range into intervals."""
        indices = torch.arange(offset, offset + num_items)  # Reproducible intervals
        groups = [indices[i:min(i + self.group_size, len(indices))] for i in range(0, len(indices), self.group_size)]
        return groups

    def __iter__(self):

        # Create groups for original and generated indices
        # generated permutation requires generated images appended after original images!
        original_groups = self._create_groups(self.num_original)
        generated_groups = self._create_groups(self.num_generated, offset=self.num_original)
        
        # Shuffle groups
        random.shuffle(original_groups)
        random.shuffle(generated_groups)

        # Ungroup indices by concatenating the groups into a single tensor
        original_grouped_perm = torch.cat(original_groups)
        generated_grouped_perm = torch.cat(generated_groups)

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, _ in enumerate(batch_starts):

            # Slicing the permutation to get batch indices, avoiding going out of bound
            original_indices = original_grouped_perm[min(i * self.num_original_batch, self.num_original) : min((i+1) * self.num_original_batch, self.num_original)]
            generated_indices = generated_grouped_perm[min(i * self.num_generated_batch, self.num_generated) : min((i+1) * self.num_generated_batch, self.num_generated)]

            # Combine
            batch_indices = torch.cat((original_indices, generated_indices))
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices.tolist()

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, original_dataset, stylized_original_dataset, generated_dataset, stylized_generated_dataset, style_mask_orig, 
                 style_mask_gen, transforms_preprocess, transforms_basic, transforms_orig_after_style, transforms_gen_after_style, 
                 transforms_orig_after_nostyle, transforms_gen_after_nostyle, robust_samples=0, group_size=32):
        self.original_dataset = original_dataset
        self.stylized_original_dataset = stylized_original_dataset
        self.generated_dataset = generated_dataset
        self.stylized_generated_dataset = stylized_generated_dataset
        self.style_mask_orig = style_mask_orig
        self.style_mask_gen = style_mask_gen
        self.preprocess = transforms_preprocess
        self.transforms_basic = transforms_basic
        self.transforms_orig_after_style = transforms_orig_after_style
        self.transforms_gen_after_style = transforms_gen_after_style
        self.transforms_orig_after_nostyle = transforms_orig_after_nostyle
        self.transforms_gen_after_nostyle = transforms_gen_after_nostyle
        self.robust_samples = robust_samples
        #self.group_size = group_size

        self.num_original = len(original_dataset) if original_dataset else 0
        self.num_generated = len(generated_dataset['images']) if generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

        # Create groups for original and generated indices like in the Batch Sampler
        #self.original_groups = self._create_groups(self.num_original)
        #self.generated_groups = self._create_groups(self.num_generated, offset=self.num_original)

        # Initialize cached dataset
        #self.cached_dataset = [None] * self.total_size
        #if robust_samples == 2:
        #    self.cached_dataset_2 = [None] * self.total_size

    def _create_groups(self, num_items, offset=0):
        """Create reproducible groups by dividing the range into intervals."""
        indices = torch.arange(offset, offset + num_items)  # Reproducible intervals
        groups = [indices[i:min(i + self.group_size, len(indices))] for i in range(0, len(indices), self.group_size)]
        return groups
    
    def _process_batch(self, group_indices, is_generated):
        """Process a batch of images and return processed images and labels."""
        if is_generated:
            # Load images and labels from the generated dataset
            images = [self.preprocess(Image.fromarray(self.generated_dataset['images'][idx - self.num_original])) for idx in group_indices]
            labels = [self.generated_dataset['labels'][idx - self.num_original] for idx in group_indices]
            images = torch.stack(images)
            images = self.transforms_batch_gen(images)
        else:
            # Load images and labels from the original dataset
            images, labels = zip(*[self.original_dataset[idx] for idx in group_indices])
            images = torch.stack([self.preprocess(img) for img in images])
            images = self.transforms_batch_orig(images)

        return images, labels

    def __getitem__(self, idx):

        if idx < self.num_original:
            x, y = self.stylized_original_dataset[idx]
            is_generated = False
            is_stylized = self.style_mask_orig[idx] if self.style_mask_orig is not None else False
        else:
            x = Image.fromarray(self.stylized_generated_dataset['images'][idx - self.num_original])
            y = self.stylized_generated_dataset['labels'][idx - self.num_original]
            is_generated = True
            is_stylized = self.style_mask_gen[idx - self.num_original] if self.style_mask_gen is not None else False


        #if self.cached_dataset[idx] is None:
#
#            # Determine group and dataset
#            group_list = self.generated_groups if is_generated else self.original_groups
#            # Find the group containing the index
#            group_idx = next(i for i, group in enumerate(group_list) if idx in group)
#            group_indices = group_list[group_idx]
#
#            # Process the batch
#            images, labels = self._process_batch(group_indices.tolist(), is_generated)
#
#            # Cache the processed group
#            for i, index in enumerate(group_indices):
#                self.cached_dataset[index] = (images[i], labels[i])
#        
#        x, y = self.cached_dataset[idx]

        if is_generated:
            aug = self.transforms_gen_after_style if is_stylized else self.transforms_gen_after_nostyle
        else:
            aug = self.transforms_orig_after_style if is_stylized else self.transforms_orig_after_nostyle

        augment = transforms.Compose([self.preprocess, self.transforms_basic, aug])
        
        if self.robust_samples == 0:
            return augment(x), y
    
        elif self.robust_samples == 1:
            if idx < self.num_original:
                x0, _ = self.original_dataset[idx]
            else:
                x0 = Image.fromarray(self.generated_dataset['images'][idx - self.num_original])

            x_tuple = (self.preprocess(x0), augment(x))
            return x_tuple, y
        
        elif self.robust_samples == 2:
            if idx < self.num_original:
                x0, _ = self.original_dataset[idx]
            else:
                x0 = Image.fromarray(self.generated_dataset['images'][idx - self.num_original])

            x_tuple = (self.preprocess(x0), augment(x), augment(x))
            return x_tuple, y

    def __len__(self):
        return self.total_size

class DataLoading():
    def __init__(self, dataset, epochs=200, generated_ratio=0.0, resize = False, run=0, test_only = False, factor = 1):
        self.dataset = dataset
        self.generated_ratio = generated_ratio
        self.resize = resize
        self.run = run
        self.epochs = epochs
        self.test_only = test_only
        self.factor = factor

    def create_transforms(self, train_aug_strat_orig, train_aug_strat_gen, RandomEraseProbability=0.0):
        # list of all data transformations used
        t = transforms.ToTensor()
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        flip = transforms.RandomHorizontalFlip()
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        c224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4)) #, value='random' --> normally distributed and out of bounds 0-1
        TAc = custom_transforms.CustomTA_color()
        TAg = custom_transforms.CustomTA_geometric()

        def stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0):
            vgg, decoder = style_transfer.load_models()
            style_feats = style_transfer.load_feat_files()
            pixels = 224 if self.dataset=='ImageNet' else 32 * self.factor

            Stylize = style_transfer.NSTTransform(style_feats, vgg, decoder, alpha_min=alpha_min, alpha_max=alpha_max, probability=probability, pixels=pixels)
            return Stylize

        def transform_not_found(train_aug_strat, dataset):
            print('Training augmentation strategy', train_aug_strat, 'could not be found. Proceeding without '
                                                                        'augmentation strategy for.', dataset, '.')
            return transforms.Compose([self.transforms_preprocess, re]), None, None

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset == 'ImageNet':
            self.transforms_preprocess = transforms.Compose([t, r256, c224])
        elif self.resize == True:
            self.transforms_preprocess = transforms.Compose([t, r224])
        else:
            self.transforms_preprocess = transforms.Compose([t])

        # standard augmentations of training set, without tensor transformation
        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset == 'TinyImageNet':
            self.transforms_basic = transforms.Compose([flip, c64])
        else:
            self.transforms_basic = transforms.Compose([flip])

        # additional transforms with tensor transformation, Random Erasing after tensor transformation

        self.transforms_orig_gpu = None
        self.transforms_gen_gpu = None

        transform_map = { #this contain a tuple each, defining the CPU transforms later applied in the dataloader and if needed GPU transforms later applied on the batch
            "TAorRE": (custom_transforms.RandomChoiceTransforms([TAc,
                                            TAg,
                                            transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                            transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                        [0.4,0.3,0.15,0.15]),
                            None,
                            None),
            "TAc+TAg+RE": (transforms.Compose([TAc, TAg, re]),
                            None,
                            None),
            "TAc+TAgorRE": (transforms.Compose([TAc,
                                        custom_transforms.RandomChoiceTransforms([TAg,
                                                                transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                                                transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                        [6, 1, 1])]),
                            None,
                            None),
            "TAc+REorTAg": (custom_transforms.RandomChoiceTransforms([TAg,
                                                    transforms.Compose([TAc, transforms.RandomErasing(p=0.525, scale=(0.02, 0.4), value='random')])],
                            [6, 8]),
                            None,
                            None),
            "StyleTransfer": transforms.Compose([stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0), re]),
            "TAorStyle0.75": transforms.Compose([custom_transforms.RandomChoiceTransforms((transforms_v2.TrivialAugmentWide(), stylization(probability=0.95)), (0.25, 0.75)), re]),
            "TAorStyle0.5": (custom_transforms.DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0)), 
                             re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
            "TAorStyle0.25": transforms.Compose([custom_transforms.RandomChoiceTransforms((transforms_v2.TrivialAugmentWide(), stylization(probability=0.95)), (0.75, 0.25)), re]),
            "TAorStyle0.1": (custom_transforms.DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0)), 
                             re,
                             transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
            "StyleAndTA": (custom_transforms.StylizedChoiceTransforms(transforms={"before_stylization": custom_transforms.EmptyTransforms(), 
                                                        "before_no_stylization": custom_transforms.EmptyTransforms()}, 
                                                        probabilities={"before_stylization_probability": 1.0, 
                                                        "before_no_stylization_probability": 0.0}),
                                stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0),
                              transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
            "weakerStyleAndTA": (custom_transforms.StylizedChoiceTransforms(transforms={"before_stylization": custom_transforms.EmptyTransforms(), 
                                                        "before_no_stylization": custom_transforms.EmptyTransforms()}, 
                                                        probabilities={"before_stylization_probability": 1.0, 
                                                        "before_no_stylization_probability": 0.0}),
                                stylization(probability=0.95, alpha_min=0.1, alpha_max=0.2),
                              transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),                                              
            "Style0.5AndTA": (custom_transforms.StylizedChoiceTransforms(transforms={"before_stylization": custom_transforms.EmptyTransforms(), 
                                                        "before_no_stylization": transforms.Compose([transforms_v2.TrivialAugmentWide(), re])}, 
                                                        probabilities={"before_stylization_probability": 0.5, 
                                                        "before_no_stylization_probability": 0.5}),
                                stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0),
                              transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
            "TrivialAugmentWide": (transforms.Compose([transforms_v2.TrivialAugmentWide(), re]),
                            None,
                            None),
            "RandAugment": (transforms.Compose([transforms_v2.RandAugment(), re]),
                            None,
                            None),
            "AutoAugment": (transforms.Compose([transforms_v2.AutoAugment(), re]),
                            None,
                            None),
            "AugMix": (transforms.Compose([transforms_v2.AugMix(), re]),
                       None,
                       None),
            'None': (None, 
                     re,
                     re),
        }

        
        self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = (transform_map[train_aug_strat_orig]
            if train_aug_strat_orig in transform_map
            else transform_not_found(train_aug_strat_orig, 'transforms_original'))

        self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = (transform_map[train_aug_strat_gen]
                                        if train_aug_strat_gen in transform_map
                                        else transform_not_found(train_aug_strat_gen, 'transforms_generated'))
        

    def load_base_data(self, validontest, run=0):

        # Trainset and Validset
        if self.test_only == False:
            if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'../data/{self.dataset}/train'))
            else:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.base_trainset = load_helper(root=os.path.abspath('../data'), train=True, download=True)

            if validontest == False:
                validsplit = 0.2
                train_indices, val_indices, _, _ = train_test_split(
                    range(len(self.base_trainset)),
                    self.base_trainset.targets,
                    stratify=self.base_trainset.targets,
                    test_size=validsplit,
                    random_state=run)  # same validation split for same runs, but new validation on multiple runs
                self.base_trainset = Subset(self.base_trainset, train_indices)
                self.validset = Subset(self.base_trainset, val_indices)
                self.validset = list(map(self.transforms_preprocess, self.validset))
            else:
                if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                    self.validset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'../data/{self.dataset}/val'),
                                                                transform=self.transforms_preprocess)
                elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                    load_helper = getattr(torchvision.datasets, self.dataset)
                    self.validset = load_helper(root=os.path.abspath('../data'), train=False, download=True,
                                           transform=self.transforms_preprocess)
                else:
                    print('Dataset not loadable')
        else:
            self.trainset = None
            self.validset = None

        #Testset
        if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
            self.testset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'../data/{self.dataset}/val'),
                                                        transform=self.transforms_preprocess)
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            load_helper = getattr(torchvision.datasets, self.dataset)
            self.testset = load_helper(root=os.path.abspath('../data'), train=False, download=True, transform=self.transforms_preprocess)
        else:
            print('Dataset not loadable')

        self.num_classes = len(self.testset.classes)
    
    def load_augmented_traindata(self, target_size, epoch=0, robust_samples=0):
        self.robust_samples = robust_samples
        self.target_size = target_size
        self.generated_dataset = np.load(os.path.abspath(f'../data/{self.dataset}-add-1m-dm.npz'),
                                    mmap_mode='r') if self.generated_ratio > 0.0 else None
        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        self.num_generated = int(target_size * self.generated_ratio)
        self.num_original = target_size - self.num_generated

        if self.num_original > 0:
            original_indices = torch.randperm(len(self.base_trainset))[:self.num_original]
            original_subset = Subset(self.base_trainset, original_indices)
            if self.stylization_orig is not None:
                stylized_original_subset, style_mask_orig = self.stylization_orig(original_subset)
            else: 
                stylized_original_subset, style_mask_orig = None, None
        else:
            original_subset, stylized_original_subset, style_mask_orig = None, None, None

        if self.num_generated > 0 and self.generated_dataset is not None:
            generated_indices = torch.randperm(len(self.generated_dataset['image']))[:self.num_generated]
            generated_subset = {
                'images': self.generated_dataset['image'][generated_indices],
                'labels': self.generated_dataset['label'][generated_indices]
            }

            if self.stylization_gen is not None:
                stylized_generated_subset, style_mask_gen = self.stylization_gen(generated_subset)
            else:
                stylized_generated_subset, style_mask_gen = None, None
        else:
            generated_subset, stylized_generated_subset, style_mask_gen = None, None, None

        self.trainset = AugmentedDataset(original_subset, stylized_original_subset, generated_subset, stylized_generated_subset, 
                                         style_mask_orig, style_mask_gen, self.transforms_preprocess, 
                                         self.transforms_basic, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                        self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples)

    def load_data_c(self, subset, subsetsize):

        c_datasets = []
        #c-corruption benchmark: https://github.com/hendrycks/robustness
        corruptions_c = np.asarray(np.loadtxt(os.path.abspath('../data/c-labels.txt'), dtype=list))

        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            corruptions_bar = np.asarray(np.loadtxt(os.path.abspath('../data/c-bar-labels-cifar.txt'), dtype=list))
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]

            for corruption, set in corruptions:
                subtestset = self.testset
                np_data_c = np.load(os.path.abspath(f'../data/{self.dataset}-{set}/{corruption}.npy'))
                np_data_c = np.array(np.array_split(np_data_c, 5))

                if subset == True:
                    np.random.seed(0)
                    selected_indices = np.random.choice(10000, subsetsize, replace=False)
                    subtestset = Subset(self.testset, selected_indices)
                    np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]

                concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, self.resize) for intensity_data_c in np_data_c])
                c_datasets.append(concat_intensities)

        elif self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            corruptions_bar = np.asarray(np.loadtxt(os.path.abspath('../data/c-bar-labels-IN.txt'), dtype=list))
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            for corruption, set in corruptions:
                intensity_datasets = [torchvision.datasets.ImageFolder(root=os.path.abspath(f'../data/{self.dataset}-{set}/' + corruption + '/' + str(intensity)),
                                                                       transform=self.transforms_preprocess) for intensity in range(1, 6)]
                if subset == True:
                    selected_indices = np.random.choice(len(intensity_datasets[0]), subsetsize, replace=False)
                    intensity_datasets = [Subset(intensity_dataset, selected_indices) for intensity_dataset in intensity_datasets]
                concat_intensities = ConcatDataset(intensity_datasets)
                c_datasets.append(concat_intensities)
        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')
            return

        if subset == True:
            c_datasets = ConcatDataset(c_datasets)
            self.c_datasets_dict = {'combined': c_datasets}
        else:
            self.c_datasets_dict = {label: dataset for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)}

        return self.c_datasets_dict

    def get_loader(self, batchsize, number_workers):
        self.number_workers = number_workers
        self.batchsize = batchsize

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        if self.generated_ratio > 0.0:
            self.CustomSampler = BalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                 batch_size=batchsize)
        else:
            self.CustomSampler = BatchSampler(RandomSampler(self.trainset), batch_size=batchsize, drop_last=False)

        self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                      num_workers=number_workers, worker_init_fn=seed_worker, 
                                        generator=g)

        self.validationloader = DataLoader(self.validset, batch_size=batchsize, pin_memory=False, num_workers=0)

        return self.trainloader, self.validationloader

    def update_trainset(self, epoch, start_epoch):

        if self.generated_ratio != 0.0 and epoch != 0 and epoch != start_epoch:
            self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples)

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True, 
                                      num_workers=self.number_workers, worker_init_fn=seed_worker,
                                      generator=g)
        return self.trainloader