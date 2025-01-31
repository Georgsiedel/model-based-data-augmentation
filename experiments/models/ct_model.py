from experiments.utils import plot_images
import torch.nn as nn
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_noise, noise_up, apply_noise_add_and_mult
from experiments.data import normalization_values
import torch

class CtModel(nn.Module):

    def __init__(self, dataset, normalized, num_classes):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.num_classes = num_classes
        self.dataset = dataset
        self.mean, self.std = normalization_values(batch=None, dataset=dataset, normalized=normalized, manifold=False,
                                                   manifold_factor=1)
        if normalized:
            self.register_buffer('mu', self.mean)
            self.register_buffer('sigma', self.std)

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x
    
    def noise_mixup(self, out, targets, robust_samples, corruptions, mixup_alpha, mixup_p, cutmix_alpha, cutmix_p, noise_minibatchsize,
                            concurrent_combinations, noise_sparsity, noise_patch_lower_scale, noise_patch_upper_scale,
                            generated_ratio):

        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -1
        else: k = 0

        if k == 0:  # Do input mixup if k is 0
            mixed_out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, generated_ratio, manifold=False, inplace=True)
            noisy_out = apply_noise(mixed_out, noise_minibatchsize, corruptions, concurrent_combinations,
                                                            self.normalized, self.dataset,
                                                            manifold=False, manifold_factor=1, noise_sparsity=noise_sparsity,
                                                            noise_patch_lower_scale=noise_patch_lower_scale,
                                                            noise_patch_upper_scale=noise_patch_upper_scale)
            out = noisy_out
            #plot_images(4, self.mean, self.std, noisy_out, noisy_out)

        return out, targets

    def forward_noise_mixup(self, out, targets, robust_samples, corruptions, mixup_alpha, mixup_p, manifold,
                            manifold_noise_factor, cutmix_alpha, cutmix_p, noise_minibatchsize,
                            concurrent_combinations, noise_sparsity, noise_patch_lower_scale, noise_patch_upper_scale,
                            generated_ratio):

        #define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False: k = -1
        elif manifold == True: k = np.random.choice(range(3), 1)[0]
        else: k = 0

        if k == 0:  # Do input mixup if k is 0
            mixed_out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, generated_ratio, manifold=False, inplace=True)
            noisy_out = apply_noise(mixed_out, noise_minibatchsize, corruptions, concurrent_combinations,
                                                            self.normalized, self.dataset,
                                                            manifold=False, manifold_factor=1, noise_sparsity=noise_sparsity,
                                                            noise_patch_lower_scale=noise_patch_lower_scale,
                                                            noise_patch_upper_scale=noise_patch_upper_scale)
            out = noisy_out
            #plot_images(4, self.mean, self.std, noisy_out, noisy_out)

        out = self.blocks[0](out)

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            out = ResidualBlock(out)
            if k == (i + 1):  # Do manifold mixup if k is greater 0
                out, targets = mixup_process(out, targets, robust_samples, self.num_classes, mixup_alpha, mixup_p,
                                         cutmix_alpha, cutmix_p, generated_ratio, manifold=True, inplace=False)
                out = noise_up(out, robust_samples=robust_samples, add_noise_level=0.5, mult_noise_level=0.5,
                                        sparse_level=0.65, l0_level=0.0)
        return out, targets

