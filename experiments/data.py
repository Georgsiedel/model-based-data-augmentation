import random
import os
import time

import torch
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, Dataset, ConcatDataset, RandomSampler, BatchSampler, Sampler, DataLoader
import numpy as np
import style_transfer
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def __init__(self, original_dataset, generated_dataset, transforms_preprocess, transforms_basic, 
                 robust_samples=0):
        self.original_dataset = original_dataset
        self.generated_dataset = generated_dataset
        self.preprocess = transforms_preprocess
        self.transforms_basic = transforms_basic
        self.robust_samples = robust_samples

        self.num_original = len(original_dataset) if original_dataset else 0
        self.num_generated = len(generated_dataset['images']) if generated_dataset else 0
        self.total_size = self.num_original + self.num_generated

    def __getitem__(self, idx):
        if idx < self.num_original:
            x, y = self.original_dataset[idx]
        else:
            # Select data from the generated dataset
            gen_idx = idx - self.num_original
            x = Image.fromarray(self.generated_dataset['images'][gen_idx])
            y = self.generated_dataset['labels'][gen_idx]

        augment = transforms.Compose([self.preprocess, self.transforms_basic])

        if self.robust_samples == 0:
            x = augment(x)
            return x, y
        elif self.robust_samples == 1:
            x1 = augment(x)
            x_tuple = (self.preprocess(x), x1)
            return x_tuple, y
        elif self.robust_samples == 2:
            x1 = augment(x)
            x2 = augment(x)
            x_tuple = (self.preprocess(x), x1, x2)
            return x_tuple, y

    def __len__(self):
        return self.total_size

class RandomChoiceTransforms:
    def __init__(self, transforms, p):
        assert len(transforms) == len(p), "The number of transforms and probabilities must match."

        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        choice = random.choices(self.transforms, self.p)[0]
        return choice(x)

class BatchedRandomChoiceTransforms:
    def __init__(self, batched_transform, batched_probability, other_transforms, other_probabilities):
        """
        Args:
            transforms: List of transform functions. One must be named "stylization".
            probabilities: List of probabilities corresponding to each transform.
        """
        assert len(other_transforms) == len(other_probabilities), "Number of transforms and probabilities must match."
        assert abs(sum(other_probabilities + [batched_probability]) - 1.0) < 1e-6, "Probabilities must sum to 1."

        # The stylization transform must be the first passed in the 
        self.batched_transform = batched_transform
        self.other_transforms = other_transforms
        self.batched_probability = batched_probability
        self.other_probabilities = other_probabilities

    def __call__(self, batch):
        """
        Args:
            batch: Tensor of shape [batch_size, ...]

        Returns:
            Transformed batch of the same shape.
        """
        batch_size = len(batch)
        device = batch.device

        # Randomly assign each image to a transform
        choices = random.choices(self.other_transforms + [self.batched_transform], self.other_probabilities + [self.batched_probability], k=batch_size)

        # Mask for batched stylization transform
        stylization_mask = torch.tensor([choice == self.batched_transform for choice in choices], device=device)
        
        # Apply stylization in a batch-wise manner
        if stylization_mask.any():
            stylized_batch = self.batched_transform(batch[stylization_mask])
        else:
            stylized_batch = torch.empty_like(batch[stylization_mask])  # Empty tensor if no stylization is applied

        # Apply other transforms iteratively
        other_transformed = torch.empty_like(batch)
        for idx, (img, choice) in enumerate(zip(batch, choices)):
            if not stylization_mask[idx]:
                other_transformed[idx] = choice(img)

        # Merge results
        result_batch = torch.clone(batch)
        result_batch[stylization_mask] = stylized_batch
        result_batch[~stylization_mask] = other_transformed[~stylization_mask]

        return result_batch

    
class EmptyTransforms:
    def __init__(self):
        pass  # No operations needed for empty transforms.

    def __call__(self, x):
        return x

class StylizedChoiceTransforms:
    def __init__(self, transforms, probabilities):
        assert len(transforms) == len(probabilities) == 2, "The number of transforms and probabilities must be 2, one before Stylization and one without Stylization."
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, x):
        choice = random.choices(list(self.transforms.items()), list(self.probabilities.values()))[0]
        type, function = choice[0], choice[1]
        if type == "before_stylization":
            return function(x), True
        elif type == "before_no_stylization":
            return function(x), False
        else:
            raise ValueError("Invalid dict key for stylized choice transform.")

class CustomTA_color(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Posterize": (lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(), False),
    "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
    "AutoContrast": (lambda num_bins, height, width: None, False),
    "Equalize": (lambda num_bins, height, width: None, False)
    }

class CustomTA_geometric(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
    }

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
        TAc = CustomTA_color()
        TAg = CustomTA_geometric()

        def stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0):
            vgg, decoder = style_transfer.load_models()
            style_feats = style_transfer.load_feat_files()
            self.stylization_prob = probability
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
            "TAorRE": (RandomChoiceTransforms([TAc,
                                            TAg,
                                            transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                            transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                        [0.4,0.3,0.15,0.15]),
                            None,
                            None),
            "TAc+TAg+RE": (transforms.Compose([CustomTA_color(), CustomTA_geometric(), re]),
                            None,
                            None),
            "TAc+TAgorRE": (transforms.Compose([CustomTA_color(),
                                        RandomChoiceTransforms([CustomTA_geometric(),
                                                                transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                                                transforms.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                        [6, 1, 1])]),
                            None,
                            None),
            "TAc+REorTAg": (RandomChoiceTransforms([TAg,
                                                    transforms.Compose([TAc, transforms.RandomErasing(p=0.525, scale=(0.02, 0.4), value='random')])],
                            [6, 8]),
                            None,
                            None),
            "StyleTransfer": transforms.Compose([stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0), re]),
            "TAorStyle0.75": transforms.Compose([RandomChoiceTransforms((transforms_v2.TrivialAugmentWide(), stylization(probability=0.95)), (0.25, 0.75)), re]),
            "TAorStyle0.5": (BatchedRandomChoiceTransforms(batched_transform=stylization(probability=0.95), batched_probability=0.5, other_transforms=[transforms_v2.TrivialAugmentWide()], other_probabilities=[0.5]), 
                             re),
            "TAorStyle0.25": transforms.Compose([RandomChoiceTransforms((transforms_v2.TrivialAugmentWide(), stylization(probability=0.95)), (0.75, 0.25)), re]),
            "TAorStyle0.1": (BatchedRandomChoiceTransforms(batched_transform=stylization(probability=0.95), batched_probability=0.1, other_transforms=[transforms_v2.TrivialAugmentWide()], other_probabilities=[0.9]), 
                             re),
            "StyleAndTA": (StylizedChoiceTransforms(transforms={"before_stylization": EmptyTransforms(), 
                                                        "before_no_stylization": EmptyTransforms()}, 
                                                        probabilities={"before_stylization_probability": 1.0, 
                                                        "before_no_stylization_probability": 0.0}),
                                stylization(probability=0.95, alpha_min=0.2, alpha_max=1.0),
                              transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
            "weakerStyleAndTA": (StylizedChoiceTransforms(transforms={"before_stylization": EmptyTransforms(), 
                                                        "before_no_stylization": EmptyTransforms()}, 
                                                        probabilities={"before_stylization_probability": 1.0, 
                                                        "before_no_stylization_probability": 0.0}),
                                stylization(probability=0.95, alpha_min=0.1, alpha_max=0.2),
                              transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),                                              
            "Style0.5AndTA": (StylizedChoiceTransforms(transforms={"before_stylization": EmptyTransforms(), 
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
                     re),
        }

        self.batch_transform_orig, self.image_transform_orig = (transform_map[train_aug_strat_orig]
            if train_aug_strat_orig in transform_map
            else transform_not_found(train_aug_strat_orig, 'transforms_original'))

        self.batch_transform_gen, self.image_transform_gen = (transform_map[train_aug_strat_gen]
                                        if train_aug_strat_gen in transform_map
                                        else transform_not_found(train_aug_strat_gen, 'transforms_generated'))
        
        #self.transforms_gpu = GPU_Transforms(self.transforms_orig_gpu, self.transforms_orig_post, self.transforms_gen_gpu, self.transforms_gen_post)

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
        else:
            original_subset = None

        if self.num_generated > 0 and self.generated_dataset is not None:
            generated_indices = torch.randperm(len(self.generated_dataset['image']))[:self.num_generated]
            generated_subset = {
                'images': self.generated_dataset['image'][generated_indices],
                'labels': self.generated_dataset['label'][generated_indices]
            }
        else:
            generated_subset = None

        self.trainset = AugmentedDataset(original_subset, generated_subset, self.transforms_preprocess, 
                                         self.transforms_basic, self.robust_samples)

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
    
    def custom_collate_fn(self, batch):

        inputs, labels = zip(*batch)
        batch_inputs = torch.stack(inputs)

        # Apply the batched random choice transform
        batch_inputs[:-int(self.generated_ratio*self.batchsize)] = self.batch_transform_orig(batch_inputs[:-int(self.generated_ratio*self.batchsize)])
        batch_inputs[-int(self.generated_ratio*self.batchsize):] = self.batch_transform_gen(batch_inputs[-int(self.generated_ratio*self.batchsize):])

        for i in range(len(batch_inputs)):
            batch_inputs[i] = self.image_transform_orig(batch_inputs[i]) if i < (len(batch_inputs)-int(self.generated_ratio*self.batchsize)) else self.image_transform_gen(batch_inputs[i])

        return batch_inputs, torch.tensor(labels)

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
                                      collate_fn=self.custom_collate_fn,
                                        generator=g)

        self.validationloader = DataLoader(self.validset, batch_size=batchsize, pin_memory=False, num_workers=0)

        return self.trainloader, self.validationloader

    def update_trainset(self, epoch, start_epoch):

        if self.generated_ratio != 0.0 and epoch != 0 and epoch != start_epoch:
            self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples)

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True,
                                      num_workers=self.number_workers, collate_fn=self.custom_collate_fn, worker_init_fn=seed_worker, generator=g)
        return self.trainloader

class GPU_Transforms():
    def __init__(self, transforms_orig_gpu, transforms_orig_post, transforms_gen_gpu, transforms_gen_post):

        self.transforms_orig_gpu = transforms_orig_gpu
        self.transforms_orig_post = transforms_orig_post
        self.transforms_gen_gpu = transforms_gen_gpu
        self.transforms_gen_post = transforms_gen_post

    def __call__(self, x, sources, apply):
        
        if self.transforms_orig_gpu == None and self.transforms_gen_gpu == None:
            return x

        x = x.to(device)

        if x.size(0) == 2 * sources.size(0):
            sources = torch.cat([sources, sources], dim=0)
        
        orig_mask = (sources) & (apply)
        if orig_mask.any():
            if apply[sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[orig_mask], 200)
                processed_chunks = [self.transforms_orig_gpu(chunk) for chunk in chunks]
                x[orig_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[orig_mask] = self.transforms_orig_gpu(x[orig_mask])
        
        gen_mask = (~sources) & (apply)
        if gen_mask.any():
            if apply[~sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[gen_mask], 200)
                processed_chunks = [self.transforms_gen_gpu(chunk) for chunk in chunks]
                x[gen_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[gen_mask] = self.transforms_gen_gpu(x[gen_mask])
        
        x = x.cpu()
        if orig_mask.any():
            x[orig_mask] = torch.stack([self.transforms_orig_post(image) for image in x[orig_mask]])
        if gen_mask.any():
            x[gen_mask] = torch.stack([self.transforms_gen_post(image) for image in x[gen_mask]])
        x = x.to(device)

        return x

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


def apply_augstrat(batch, train_aug_strat):

    for id, img in enumerate(batch):
        img = img * 255.0
        img = img.type(torch.uint8)
        tf = getattr(transforms, train_aug_strat)
        img = tf()(img)
        img = img.type(torch.float32) / 255.0
        batch[id] = img

    return batch
