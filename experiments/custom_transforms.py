import random
import torch
import torch.cuda.amp
import torch.utils
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms
from run_exp import device
import gc

import torch
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
import time
from experiments.utils import plot_images
import experiments.style_transfer as style_transfer
from experiments.data import StylizedTensorDataset

def get_transforms_map(strat_name, re, dataset, factor):
    TAc = CustomTA_color()
    TAg = CustomTA_geometric()

    def stylization(dataset, factor, probability=0.95, alpha_min=0.2, alpha_max=1.0):
        def lazy():
            vgg, decoder = style_transfer.load_models()
            style_feats = style_transfer.load_feat_files()
            stylization_prob = probability
            pixels = 224 if dataset=='ImageNet' else 32 * factor

            return style_transfer.NSTTransform(style_feats, vgg, decoder, alpha_min=alpha_min, alpha_max=alpha_max, probability=stylization_prob, pixels=pixels)
        return lazy

    transform_map = { #this contain a tuple each, defining the CPU transforms later applied in the dataloader and if needed GPU transforms later applied on the batch
        "StyleTransferalpha00": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.0, alpha_max=0.0)), 
                            re,
                            re),    
        "StyleTransferalpha02": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.2, alpha_max=0.2)), 
                            re,
                            re), 
        "StyleTransferalpha04": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.4, alpha_max=0.4)), 
                            re,
                            re),    
        "StyleTransferalpha06": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.6, alpha_max=0.6)), 
                            re,
                            re), 
        "StyleTransferalpha08": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.8, alpha_max=0.8)), 
                            re,
                            re), 
        "StyleTransferalpha10": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re), 
        "StyleTransfer100": (DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer90": (DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer80": (DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer70": (DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer60": (DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),     
        "StyleTransfer50": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),    
        "StyleTransfer50alpha01-10": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.1, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer50alpha05-10": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            re),  
        "StyleTransfer50alpha04-07": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.4, alpha_max=0.7)), 
                            re,
                            re),
        "StyleTransfer50alpha05-08": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.5, alpha_max=0.8)), 
                            re,
                            re),
        "StyleTransfer50alpha08-10": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=0.8, alpha_max=1.0)), 
                            re,
                            re),
        "StyleTransfer40": (DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),     
        "StyleTransfer30": (DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),     
        "StyleTransfer20": (DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),     
        "StyleTransfer10": (DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=stylization(dataset, factor, probability=1.0, alpha_min=1.0, alpha_max=1.0)), 
                            re,
                            re),                                                               
        "TAorStyle90": (DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle80": (DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                        transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle70": (DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle60": (DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle50": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle40": (DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                        transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle30": (DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle20": (DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "TAorStyle10": (DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            re,
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
        "StyleAndTA": (DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re]),
                            transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),               
        "StyleorColorAndGeometricOrRE": (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=stylization(dataset, factor, probability=0.95, alpha_min=0.5, alpha_max=1.0)), 
                            RandomChoiceTransforms([TAg,
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                    [0.7,0.15,0.15]),
                            transforms.Compose([TAc, RandomChoiceTransforms([TAg,
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                    [0.7,0.15,0.15])])),                                  
        "TAorRE": (None,
                        None,
                        RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(),
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'),
                                        transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)],
                                    [0.8,0.1,0.1])),
        "TrivialAugmentWide": (None,
                        None,
                        transforms.Compose([transforms_v2.TrivialAugmentWide(), re])),
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
    
    def transform_not_found(name):
        print('Training augmentation strategy', name, 'could not be found. Proceeding without augmentation strategy.')
        return None, re, re

    transforms_stylization, transforms_after_style, transforms_after_nostyle = (transform_map[strat_name]
    if strat_name in transform_map
    else transform_not_found(strat_name))

    return transforms_stylization, transforms_after_style, transforms_after_nostyle

class DatasetStyleTransforms:
    def __init__(self, transform_style, batch_size, stylized_ratio):
        """
        Args:
            transform_style: Callable to apply stylization.
            batch_size: Batch size for tensors passed to transform_style.
            stylized_ratio: Fraction of images to stylize (0 to 1).
        """
        self.transform_style = transform_style
        self.batch_size = batch_size
        self.stylized_ratio = stylized_ratio
        self.is_lazy_loaded = False


    def __call__(self, dataset):
        """
        Stylize a fraction of images in the dataset and return a new dataset.

        Args:
            dataset: PyTorch Dataset to process.

        Returns:
            stylized_dataset: A new TensorDataset with stylized images.
        """
        if not self.is_lazy_loaded and callable(self.transform_style): #for calling the transform from a dictionary
            self.transform_style = self.transform_style()
            self.is_lazy_loaded = True

        num_images = len(dataset)
        num_stylized = int(num_images * self.stylized_ratio)
        stylized_indices = torch.randperm(num_images)[:num_stylized]
        
        # Create a Subset with the stylized indices
        stylized_subset = Subset(dataset, stylized_indices)

        # DataLoader for processing the stylized subset
        loader = DataLoader(stylized_subset, batch_size=self.batch_size, shuffle=False)
        
        # Use zeros as placeholders for non-stylized images and labels
        sample_image, _ = dataset[0]  # Get sample shape from the dataset
        stylized_images = torch.zeros((num_stylized, *sample_image.shape), dtype=sample_image.dtype)

        # Iterate over the DataLoader and process stylized images
        for batch_indices, (images, _) in zip(loader.batch_sampler, loader):  
            # Apply the transformation to the batch
            transformed_images = self.transform_style(images)

            # Store the transformed images and labels in their original positions
            stylized_images[batch_indices] = transformed_images

        # Delete intermediary variables to save memory
        del loader, stylized_subset
        gc.collect()

        style_mask = torch.zeros(num_images, dtype=torch.bool)
        style_mask[stylized_indices] = True
        style_mask = style_mask.tolist()

        # Return the stylized dataset
        return StylizedTensorDataset(dataset, stylized_images, stylized_indices), style_mask

class DatasetStyleTransforms_old:
    def __init__(self, stylized_ratio, batch_size, transform_style):
        """
        Initializes the DatasetStyleTransformer.

        Args:
            stylized_ratio: Ratio of images to stylize.
            batch_size: Batch size for applying transformations.
            transform_style: Transformation to apply.
        """
        self.stylized_ratio = stylized_ratio
        self.batch_size = batch_size
        self.transform_style = transform_style
        self.is_lazy_loaded = False

    def __call__(self, dataset):
        """
        Transforms a dataset by applying style transformations to a specified ratio of images.

        Args:
            dataset: Dictionary with 'images' and 'labels' (numpy-style) or PyTorch Subset.

        Returns:
            updated_dataset: Updated dataset with stylized images.
            transformed_flags: Boolean list indicating which images were transformed.
        """
        if not self.is_lazy_loaded and callable(self.transform_style):
            self.transform_style = self.transform_style()
            self.is_lazy_loaded = True

        if isinstance(dataset, Subset):
            # Handle PyTorch Subset
            subset_indices = dataset.indices

            # Load all images at once (batch-wise slicing if supported by dataset)
            if hasattr(dataset.dataset, 'data') and hasattr(dataset.dataset, 'targets'):
                # CIFAR-like datasets (data and targets are attributes)
                images = [dataset.dataset.data[i] for i in subset_indices]  # Convert to PIL
                labels = [dataset.dataset.targets[i] for i in subset_indices]
            else:
                images = [dataset[i][0] for i in range(len(dataset.indices))]  # Images
                labels = [dataset[i][1] for i in range(len(dataset.indices))]  # Labels

        elif isinstance(dataset, dict) and 'images' in dataset and 'labels' in dataset:
            # Handle numpy-style dataset
            images = dataset['images']
            labels = dataset['labels']
        else:
            raise ValueError("Unsupported dataset format. Must be PyTorch Subset or dict with 'images' and 'labels'.")
        
        num_images = len(images)
        stylized_indices = torch.randperm(num_images)[:int(num_images * self.stylized_ratio)]

        # Process images in batches
        stylized_images = []
        for i in range(0, len(stylized_indices), self.batch_size):
            batch_indices = stylized_indices[i:min(i + self.batch_size, len(stylized_indices))]
            batch_images = [ToTensor()(images[j]) for j in batch_indices]
            batch_tensor = torch.stack(batch_images)
            transformed_batch = self.transform_style(batch_tensor)
            stylized_images.append(transformed_batch)

        # Concatenate processed images
        stylized_images = torch.cat(stylized_images)
 
        # Generate the transformed flags
        style_mask = torch.zeros(num_images, dtype=torch.bool)
        style_mask[stylized_indices] = True
        style_mask = style_mask.tolist()

        # Convert back to original format (e.g., PIL or numpy array)
        if isinstance(dataset, Subset):
            stylized_images = (stylized_images*255).to(torch.uint8).permute(0, 2, 3, 1)
            for idx, stylized_image in zip(stylized_indices, stylized_images):
                images[idx] = stylized_image.numpy()
            updated_dataset = [(img, label) for img, label in zip(images, labels)]

        elif isinstance(dataset, dict):
            stylized_images = (stylized_images*255).to(torch.uint8).permute(0, 2, 3, 1)
            updated_images = [img.numpy() for img in stylized_images]
            images[stylized_indices] = updated_images
            updated_dataset = {'images': images,
                               'labels': labels}
        else:
            raise ValueError("Unsupported dataset format. Must be PyTorch Subset or dict with 'images' and 'labels'.")
        # Return updated dataset and transformation flags

        return updated_dataset, style_mask

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