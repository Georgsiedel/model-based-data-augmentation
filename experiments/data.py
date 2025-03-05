import random
import os
import time
import gc

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset, RandomSampler, BatchSampler, Sampler, DataLoader
import numpy as np
import experiments.custom_transforms as custom_transforms
from run_exp import device
from experiments.utils import plot_images

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
    
class GeneratedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label
    
class StylizedTensorDataset(Dataset):
    def __init__(self, dataset, stylized_images, stylized_indices):
        """
        A dataset class that maps indices of the original dataset to stylized data when available.

        Args:
            dataset (torchvision.dataset): original dataset
            stylized_images (torch.Tensor): Tensor of stylized images.
            stylized_labels (torch.Tensor): Tensor of stylized labels.
            stylized_indices (list[int]): List of indices in the original dataset that correspond to stylized data.
        """
        self.dataset = dataset
        self.stylized_images = stylized_images

        # Map original dataset indices to the stylized dataset ensures efficient O(1) lookup
        self.index_map = {orig_idx.item(): i for i, orig_idx in enumerate(stylized_indices)} 

    def __len__(self):
        return len(self.dataset)
        
    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        x, _ = self.dataset[idx]
        return x

    def __getitem__(self, idx):
        if idx in self.index_map:
            # Fetch data from the stylized dataset
            stylized_idx = self.index_map[idx]
            x = self.stylized_images[stylized_idx]
            _, y = self.dataset[idx]
        else:
            x, y = self.dataset[idx]
            # Fetch data from the original dataset
        return x, y

class SubsetWithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def getclean(self, idx):#for robust loss, called in AugmentedDataset class
        image, _ = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomDataset(Dataset):
    def __init__(self, np_images, original_dataset, resize, preprocessing):
        # Load images
        self.np_images = np.memmap(np_images, dtype=np.float32, mode='r') if isinstance(np_images, str) else np_images
        self.resize = resize
        self.preprocessing = preprocessing

        # Extract labels from the original PyTorch dataset
        self.labels = [label for _, label in original_dataset]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.preprocessing(self.np_images[index])
        if self.resize == True:
            image = transforms.Resize(224, antialias=True)(image)

        label = self.labels[index]

        return image, label

class ReproducibleBalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size, epoch):
        super(ReproducibleBalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)
        self.current_epoch = epoch

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch

    def generate_indices_order(self, num_samples, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices = list(range(num_samples))
        local_rng.shuffle(indices)
        return indices

    def __iter__(self):

        # Create a single permutation for the whole epoch which is reproducible.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm = self.generate_indices_order(self.num_original, epoch=self.current_epoch)
        generated_perm = self.generate_indices_order(self.num_generated, epoch=self.current_epoch)
        self.current_epoch += 1

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, start in enumerate(batch_starts):

            # Slicing the permutation to get batch indices, avoiding going out of bound
            original_indices = original_perm[min(i * self.num_original_batch, self.num_original) : min((i+1) * self.num_original_batch, self.num_original)]
            generated_indices = generated_perm[min(i * self.num_generated_batch, self.num_generated) : min((i+1) * self.num_generated_batch, self.num_generated)]

            # Combine
            batch_indices = original_indices + generated_indices
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

class GroupedAugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, original_dataset, generated_dataset, 
                 transforms_basic, transforms_batch_gen, transforms_batch_orig, transforms_iter_orig_after_style, transforms_iter_gen_after_style,
                 transforms_iter_orig_after_nostyle, transforms_iter_gen_after_nostyle, robust_samples=0, epoch=0):
        
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.transforms_basic = transforms_basic
        self.transforms_batch_gen = transforms_batch_gen
        self.transforms_batch_orig = transforms_batch_orig
        self.transforms_iter_orig_after_style = transforms_iter_orig_after_style
        self.transforms_iter_gen_after_style = transforms_iter_gen_after_style
        self.transforms_iter_orig_after_nostyle = transforms_iter_orig_after_nostyle
        self.transforms_iter_gen_after_nostyle = transforms_iter_gen_after_nostyle

        self.robust_samples = robust_samples

        # Compute cache sizes (i.e. block sizes) based on the batch transform parameters.
        if transforms_batch_gen:
            self.cache_size_gen = int(transforms_batch_gen.batch_size / transforms_batch_gen.stylized_ratio)
        else:
            self.cache_size_gen = 1
        if transforms_batch_orig:
            self.cache_size_orig = int(transforms_batch_orig.batch_size / transforms_batch_orig.stylized_ratio)
        else:
            self.cache_size_orig = 1

        self.num_original = len(original_dataset) if original_dataset else 0
        self.num_generated = len(generated_dataset) if generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

        # Initialize empty caches. They map the global (domain) index to (image, label, style_flag).
        self.cache_orig = {}
        self.cache_gen = {}

        # Generate reproducible permutation lists for each domain.
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        """
        At the beginning of each epoch, regenerate the random ordering for each domain and clear caches.
        """
        self.original_perm = self.generate_indices_order(self.num_original, epoch)
        self.generated_perm = self.generate_indices_order(self.num_generated, epoch)
        self.cache_orig.clear()
        self.cache_gen.clear()
        
    def generate_indices_order(self, num_samples, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices = list(range(num_samples))
        local_rng.shuffle(indices)
        return indices
    
    def __getitem__(self, idx):
        """
        Retrieve the (transformed) item corresponding to a global index.
        
        For original images, the global index is used as is; for generated images,
        the index is adjusted by subtracting num_original. If the requested item is not
        in the cache, the cache is cleared and filled by processing a block (of size cache_size)
        from the corresponding permutation starting at the requested index’s position.
        Then, an iterative transform (after the batch transform) is applied based on the style flag.
        """
        # Determine domain.
        if idx < self.num_original:
            global_index = idx  # for original images
            perm = self.original_perm
            cache = self.cache_orig
            cache_size = self.cache_size_orig
            dataset = self.original_dataset
            transform_batch = self.transforms_batch_orig
            transforms_iter_after_style = self.transforms_iter_orig_after_style
            transforms_iter_after_nostyle = self.transforms_iter_orig_after_nostyle
        else:
            global_index = idx - self.num_original  # for generated images, adjust index
            perm = self.generated_perm
            cache = self.cache_gen
            cache_size = self.cache_size_gen
            dataset = self.generated_dataset
            transform_batch = self.transforms_batch_gen
            transforms_iter_after_style = self.transforms_iter_gen_after_style
            transforms_iter_after_nostyle = self.transforms_iter_gen_after_nostyle

        # If the requested global index is cached, retrieve it.
        if global_index not in cache:

            # Not in cache. Find the position of this global index in the permutation.
            try:
                pos = perm.index(global_index)
            except ValueError:
                pos = 0
            # Get the block of indices: from the found position up to cache_size items.
            indices_block = perm[pos: pos + cache_size]

            items = [dataset[i] for i in indices_block]
            images, labels = zip(*items)
            images = torch.stack(images)

            images, style_mask = transform_batch(images)

            # Clear the cache and fill it with the new block.
            cache.clear()

            for i, d_idx in enumerate(indices_block):
                cache[d_idx] = (images[i], labels[i], style_mask[i])
            
        
        x, y, style_flag = cache[global_index]

        # Apply the iterative (per-image) transform based on whether the image was styled.
        transform_iter = (transforms_iter_after_style if style_flag else transforms_iter_after_nostyle)
        
        aug = transforms.Compose([self.transforms_basic, transform_iter])

        # Handle robust_samples if needed.
        if self.robust_samples == 0:
            return aug(x), y
        
        elif self.robust_samples == 1:
            x0, _ = dataset[global_index]
            return (x0, aug(x)), y
        
        elif self.robust_samples == 2:
            x0, _ = dataset[global_index]
            return (x0, aug(x), aug(x)), y

    def __len__(self):
        return self.total_size

class BalancedRatioSampler(Sampler):
    def __init__(self, dataset, generated_ratio, batch_size):
        super(BalancedRatioSampler, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.generated_ratio = generated_ratio
        self.size = len(dataset)

        self.num_generated = int(self.size * self.generated_ratio)
        self.num_original = self.size - self.num_generated
        self.num_generated_batch = int(self.batch_size * self.generated_ratio)
        self.num_original_batch = self.batch_size - self.num_generated_batch

    def __iter__(self):

        # Create a single permutation for the whole epoch.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm = torch.randperm(self.num_original)
        generated_perm = torch.randperm(self.num_generated) + self.num_original

        batch_starts = range(0, self.size, self.batch_size)  # Start points for each batch
        for i, start in enumerate(batch_starts):

            # Slicing the permutation to get batch indices, avoiding going out of bound
            original_indices = original_perm[min(i * self.num_original_batch, self.num_original) : min((i+1) * self.num_original_batch, self.num_original)]
            generated_indices = generated_perm[min(i * self.num_generated_batch, self.num_generated) : min((i+1) * self.num_generated_batch, self.num_generated)]

            # Combine
            batch_indices = torch.cat((original_indices, generated_indices))
            #batch_indices = batch_indices[torch.randperm(batch_indices.size(0))]

            yield batch_indices.tolist()

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size


class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform augmentations and allow robust loss functions."""

    def __init__(self, stylized_original_dataset, stylized_generated_dataset, style_mask, 
                 transforms_basic, transforms_orig_after_style, transforms_gen_after_style, 
                 transforms_orig_after_nostyle, transforms_gen_after_nostyle, robust_samples=0):
        self.stylized_original_dataset = stylized_original_dataset
        self.stylized_generated_dataset = stylized_generated_dataset
        self.style_mask = style_mask
        self.transforms_basic = transforms_basic
        self.transforms_orig_after_style = transforms_orig_after_style
        self.transforms_gen_after_style = transforms_gen_after_style
        self.transforms_orig_after_nostyle = transforms_orig_after_nostyle
        self.transforms_gen_after_nostyle = transforms_gen_after_nostyle
        self.robust_samples = robust_samples

        self.num_original = len(stylized_original_dataset) if stylized_original_dataset else 0
        self.num_generated = len(stylized_generated_dataset) if stylized_generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

        assert len(style_mask) == self.num_original + self.num_generated

    def __getitem__(self, idx):

        is_stylized = self.style_mask[idx]

        if idx < self.num_original:
            x, y = self.stylized_original_dataset[idx]
            aug = self.transforms_orig_after_style if is_stylized else self.transforms_orig_after_nostyle
        else:
            x, y = self.stylized_generated_dataset[idx - self.num_original]
            aug = self.transforms_gen_after_style if is_stylized else self.transforms_gen_after_nostyle

        augment = transforms.Compose([self.transforms_basic, aug])

        if self.robust_samples == 0:
            return augment(x), int(y)
    
        elif self.robust_samples >= 1:
            if idx < self.num_original:
                x0 = self.stylized_original_dataset.getclean(idx)
            else:
                x0 = self.stylized_generated_dataset.getclean(idx - self.num_original)

            if self.robust_samples == 1:
                return (x0, augment(x)), int(y)
            elif self.robust_samples == 2:
                return (x0, augment(x), augment(x)), int(y)

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
        c224 = transforms.RandomCrop(224, padding=28)
        flip = transforms.RandomHorizontalFlip()
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        cc224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4)) #, value='random' --> normally distributed and out of bounds 0-1

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset == 'ImageNet':
            self.transforms_preprocess = transforms.Compose([t, r256, cc224])
        elif self.resize == True:
            self.transforms_preprocess = transforms.Compose([t, r224])
        else:
            self.transforms_preprocess = transforms.Compose([t])

        # standard augmentations of training set, without tensor transformation
        if self.dataset == 'ImageNet':
            self.transforms_basic = transforms.Compose([flip])
        elif self.resize:
            self.transforms_basic = transforms.Compose([flip, c224])
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset == 'TinyImageNet':
            self.transforms_basic = transforms.Compose([flip, c64])

        self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_orig, re, self.dataset, self.factor)
        self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_gen, re, self.dataset, self.factor)

    def load_base_data(self, validontest, run=0):

        # Trainset and Validset
        if self.test_only == False:
            if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'../data/{self.dataset}/train'))
            else:
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath('../data'), train=True, download=True)

            if validontest == False:
                validsplit = 0.2
                train_indices, val_indices, _, _ = train_test_split(
                    range(len(base_trainset)),
                    base_trainset.targets,
                    stratify=base_trainset.targets,
                    test_size=validsplit,
                    random_state=run)  # same validation split for same runs, but new validation on multiple runs
                self.base_trainset = Subset(base_trainset, train_indices)
                validset = Subset(base_trainset, val_indices)
                self.validset = [(self.transforms_preprocess(data), target) for data, target in validset]
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
                self.base_trainset = base_trainset
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
            original_subset = SubsetWithTransform(Subset(self.base_trainset, original_indices), self.transforms_preprocess)

            #if self.stylization_orig is not None:
            #    stylized_original_subset, style_mask_orig = self.stylization_orig(original_subset)
            #else: 
            #    stylized_original_subset, style_mask_orig = original_subset, [False] * len(original_subset)
        else:
            original_subset = None #stylized_original_subset, style_mask_orig = None, []
        
        if self.num_generated > 0 and self.generated_dataset is not None:
            generated_indices = np.random.choice(len(self.generated_dataset['label']), size=self.num_generated, replace=False)

            generated_subset = GeneratedDataset(
                self.generated_dataset['image'][generated_indices],
                self.generated_dataset['label'][generated_indices],
                transform=self.transforms_preprocess
            )

            #if self.stylization_gen is not None:
            #    stylized_generated_subset, style_mask_gen = self.stylization_gen(generated_subset)
            #else:
            #    stylized_generated_subset, style_mask_gen = generated_subset, [False] * len(generated_subset)
        else:
            generated_subset = None #stylized_generated_subset, style_mask_gen = None, []

        #style_mask = style_mask_orig + style_mask_gen
        
        self.trainset = GroupedAugmentedDataset(original_subset, generated_subset, self.transforms_basic, self.stylization_orig, 
                                self.stylization_gen, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples, epoch)

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
                np_data_c = np.load(os.path.abspath(f'../data/{self.dataset}-{set}/{corruption}.npy'), mmap_mode='r')
                np_data_c = np.array(np.array_split(np_data_c, 5))

                if subset == True:
                    np.random.seed(0)
                    selected_indices = np.random.choice(10000, subsetsize, replace=False)
                    subtestset = Subset(self.testset, selected_indices)
                    np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]
                concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, self.resize, self.transforms_preprocess) for intensity_data_c in np_data_c])
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

        #if self.generated_ratio > 0.0:
        self.CustomSampler = ReproducibleBalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                 batch_size=batchsize, epoch=self.epoch)
        #else:
        #    self.CustomSampler = BatchSampler(RandomSampler(self.trainset), batch_size=batchsize, drop_last=False)

        self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                      num_workers=number_workers, worker_init_fn=seed_worker, 
                                        generator=g, persistent_workers=False)

        self.validationloader = DataLoader(self.validset, batch_size=batchsize, pin_memory=False, num_workers=0)

        return self.trainloader, self.validationloader

    def update_trainset(self, epoch, start_epoch):

        if (self.generated_ratio != 0.0) and epoch != 0 and epoch != start_epoch:
                        
            #del self.trainset
            self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples)

        elif (self.stylization_gen is not None or self.stylization_orig is not None) and epoch != 0 and epoch != start_epoch:
                        
            #del self.trainset
            self.trainset.set_epoch(epoch)
        
        #del self.trainloader
        #gc.collect()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True, 
                                      num_workers=self.number_workers, worker_init_fn=seed_worker,
                                      generator=g, persistent_workers=False)
        return self.trainloader