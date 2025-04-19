import random
import torch
import os
from PIL import Image
import torch.cuda.amp
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler, DataLoader
import numpy as np
from run_exp import device

def custom_collate_fn(batch, batch_transform_orig, batch_transform_gen, image_transform_orig, 
                      image_transform_gen, generated_ratio, batchsize):

    inputs, labels = zip(*batch)
    batch_inputs = torch.stack(inputs)

    # Apply the batched random choice transform
    batch_inputs[:-int(generated_ratio*batchsize)] = batch_transform_orig(batch_inputs[:-int(generated_ratio*batchsize)])
    batch_inputs[-int(generated_ratio*batchsize):] = batch_transform_gen(batch_inputs[-int(generated_ratio*batchsize):])

    for i in range(len(batch_inputs)):
        batch_inputs[i] = image_transform_orig(batch_inputs[i]) if i < (len(batch_inputs)-int(generated_ratio*batchsize)) else image_transform_gen(batch_inputs[i])

    return batch_inputs, torch.tensor(labels)

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
    
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        return self.data[index]
    def __len__(self):
        return len(self.data)
    
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
    def __init__(self, np_images, testset, resize, preprocessing):
        # Load images
        self.np_images = np.memmap(np_images, dtype=np.float32, mode='r') if isinstance(np_images, str) else np_images
        self.resize = resize
        self.preprocessing = preprocessing
        self.set = testset

    def __len__(self):
        return len(self.set)

    def __getitem__(self, index):
        # Get image and label for the given index
        image = self.preprocessing(self.np_images[index])
        if self.resize == True:
            image = transforms.Resize(224, antialias=True)(image)

        _, label = self.set[index]

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
        
    def generate_indices_order(self, num_original, num_generated, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices_original = list(range(num_original))
        indices_generated = list(range(num_original, num_generated + num_original))

        local_rng.shuffle(indices_original)
        local_rng.shuffle(indices_generated)

        return indices_original, indices_generated

    def __iter__(self):

        # Create a single permutation for the whole epoch which is reproducible.
        # generated permutation requires generated images appended to the back of the dataset!
        original_perm, generated_perm = self.generate_indices_order(self.num_original, self.num_generated, self.current_epoch)
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
        self.original_perm, self.generated_perm = self.generate_indices_order(self.num_original, self.num_generated, epoch)
        self.cache_orig.clear()
        self.cache_gen.clear()
        
    def generate_indices_order(self, num_original, num_generated, epoch):
        # Use a local RNG instance that won’t disturb your global seeds.
        local_rng = random.Random(epoch)
        indices_original = list(range(num_original))
        indices_generated = list(range(num_original, num_generated + num_original))

        local_rng.shuffle(indices_original)
        local_rng.shuffle(indices_generated)

        return indices_original, indices_generated
    
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
            dataset_specific_index = idx  # for original images
            perm = self.original_perm
            cache = self.cache_orig
            cache_size = self.cache_size_orig
            dataset = self.original_dataset
            transform_batch = self.transforms_batch_orig
            transforms_iter_after_style = self.transforms_iter_orig_after_style
            transforms_iter_after_nostyle = self.transforms_iter_orig_after_nostyle
        else:
            dataset_specific_index = idx - self.num_original  # for generated images, adjust index
            perm = self.generated_perm
            cache = self.cache_gen
            cache_size = self.cache_size_gen
            dataset = self.generated_dataset
            transform_batch = self.transforms_batch_gen
            transforms_iter_after_style = self.transforms_iter_gen_after_style
            transforms_iter_after_nostyle = self.transforms_iter_gen_after_nostyle

        if transform_batch == None:
            x, y = dataset[dataset_specific_index]
            
        else:
            # If the requested global index is cached, retrieve it.
            if idx not in cache:

                # Not in cache. Find the position of this global index in the permutation.
                try:
                    pos = perm.index(idx)
                except ValueError:
                    pos = 0
                # Get the block of indices: from the found position up to cache_size items.
                indices_block = perm[pos: pos + cache_size]

                items = [dataset[i - self.num_original] for i in indices_block]
                images, labels = zip(*items)
                images = torch.stack(images)

                images, style_mask = transform_batch(images)

                # Clear the cache and fill it with the new block.
                cache.clear()

                for i, d_idx in enumerate(indices_block):
                    cache[d_idx] = (images[i], labels[i], style_mask[i])
            
            x, y, style_flag = cache[idx]

        # Apply the iterative (per-image) transform based on whether the image was styled.
        transform_iter = (transforms_iter_after_style if style_flag else transforms_iter_after_nostyle)
        
        aug = transforms.Compose([self.transforms_basic, transform_iter])

        # Handle robust_samples if needed.
        if self.robust_samples == 0:
            return aug(x), y
        
        elif self.robust_samples == 1:
            x0, _ = dataset[dataset_specific_index]
            return (x0, aug(x)), y
        
        elif self.robust_samples == 2:
            x0, _ = dataset[dataset_specific_index]
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

class StyleDataset(Dataset):
    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, file)
            for file in os.listdir(root_dir)
            if file.endswith(".jpg")
        ]
        if dataset_type in ["CIFAR10", "CIFAR100"]:
            self.transform = transforms.Resize((32, 32), antialias=True)
        elif dataset_type == "TinyImageNet":
            self.transform = transforms.Resize((64, 64), antialias=True)
        elif dataset_type == "ImageNet":
            self.transform = transforms.Resize((224, 224), antialias=True)
        else:
            raise AttributeError(f"Dataset: {dataset_type} is an unrecognized dataset")
        self.transform = transforms.Compose([self.transform, transforms.ToTensor()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image