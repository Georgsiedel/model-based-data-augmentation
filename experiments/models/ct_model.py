from experiments.utils import plot_images
import torch.nn as nn
import torch
import numpy as np
from experiments.mixup import mixup_process
from experiments.noise import apply_noise, noise_up, apply_noise_add_and_mult
from experiments.data import normalization_values
from experiments.deepaugment_n2n import N2N_DeepAugment


class CtModel(nn.Module):
    def __init__(self, dataset, normalized, num_classes):
        super(CtModel, self).__init__()
        self.normalized = normalized
        self.num_classes = num_classes
        self.dataset = dataset
        self.mean, self.std = normalization_values(
            batch=None,
            dataset=dataset,
            normalized=normalized,
            manifold=False,
            manifold_factor=1,
        )
        if normalized:
            self.register_buffer("mu", self.mean)
            self.register_buffer("sigma", self.std)

        self.deepaugment_instance = None

    def forward_normalize(self, x):
        if self.normalized:
            x = (x - self.mu) / self.sigma
        return x

    def noise_mixup(
        self,
        out,
        targets,
        robust_samples,
        corruptions,
        mixup_alpha,
        mixup_p,
        cutmix_alpha,
        cutmix_p,
        noise_minibatchsize,
        concurrent_combinations,
        noise_sparsity,
        noise_patch_lower_scale,
        noise_patch_upper_scale,
        generated_ratio,
        n2n_deepaugment,
    ):
        # define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False:
            k = -1
        else:
            k = 0

        if k == 0:  # Do input mixup if k is 0
            if n2n_deepaugment:  # apply deepaugment if True
                if self.deepaugment_instance is None:
                    self.deepaugment_instance = N2N_DeepAugment(
                        orig_batch_size=out.shape[0],
                        image_size=out.shape[2],
                        channels=out.shape[1],
                        noisenet_max_eps=0.75,
                        ratio=0.5,
                    )
                out = self.deepaugment_instance(out)
            mixed_out, targets = mixup_process(
                out,
                targets,
                robust_samples,
                self.num_classes,
                mixup_alpha,
                mixup_p,
                cutmix_alpha,
                cutmix_p,
                generated_ratio,
                manifold=False,
                inplace=True,
            )
            noisy_out = apply_noise(
                mixed_out,
                noise_minibatchsize,
                corruptions,
                concurrent_combinations,
                self.normalized,
                self.dataset,
                manifold=False,
                manifold_factor=1,
                noise_sparsity=noise_sparsity,
                noise_patch_lower_scale=noise_patch_lower_scale,
                noise_patch_upper_scale=noise_patch_upper_scale,
            )
            out = noisy_out
            # plot_images(4, self.mean, self.std, noisy_out, noisy_out)

        return out, targets

    def forward_noise_mixup(
        self,
        out,
        targets,
        robust_samples,
        corruptions,
        mixup_alpha,
        mixup_p,
        manifold,
        manifold_noise_factor,
        cutmix_alpha,
        cutmix_p,
        noise_minibatchsize,
        concurrent_combinations,
        noise_sparsity,
        noise_patch_lower_scale,
        noise_patch_upper_scale,
        generated_ratio,
        n2n_deepaugment,
        style_feats,
        **kwargs,
    ):
        if style_norm_type := kwargs.get("norm_type", None):
            int_adain_probability = kwargs.get("style_probability", 0.0)

        # define where mixup is applied. k=0 is in the input space, k>0 is in the embedding space (manifold mixup)
        if self.training == False:
            k = -1
        elif manifold == True:
            k = np.random.choice(range(3), 1)[0]
        else:
            k = 0

        if k == 0:  # Do deepaugemtn, input mixup and noise injection if k is 0
            if n2n_deepaugment:  # apply deepaugment if True
                if self.deepaugment_instance is None:
                    self.deepaugment_instance = N2N_DeepAugment(
                        orig_batch_size=out.shape[0],
                        image_size=out.shape[2],
                        channels=out.shape[1],
                        noisenet_max_eps=0.75,
                        ratio=0.5,
                    )
                out = self.deepaugment_instance(out)
            mixed_out, targets = mixup_process(
                out,
                targets,
                robust_samples,
                self.num_classes,
                mixup_alpha,
                mixup_p,
                cutmix_alpha,
                cutmix_p,
                generated_ratio,
                manifold=False,
                inplace=True,
            )
            noisy_out = apply_noise(
                mixed_out,
                noise_minibatchsize,
                corruptions,
                concurrent_combinations,
                self.normalized,
                self.dataset,
                manifold=False,
                manifold_factor=1,
                noise_sparsity=noise_sparsity,
                noise_patch_lower_scale=noise_patch_lower_scale,
                noise_patch_upper_scale=noise_patch_upper_scale,
            )
            out = noisy_out
            # plot_images(4, self.mean, self.std, noisy_out, noisy_out)

        out = self.blocks[0](out)

        if style_feats is not None:
            print("Style feats shape: ", style_feats.shape)
            style_feats = self.blocks[0](style_feats)

        prob = torch.rand(1).item()

        for i, ResidualBlock in enumerate(self.blocks[1:]):
            if style_norm_type == "pono":
                if prob < int_adain_probability and i == 0:
                    out = self.pono(out, style_feats)

            out = ResidualBlock(out)

            if style_norm_type == "int_adain":
                if prob < int_adain_probability and i == 0:
                    print(
                        f"Applying int_adain. Shape of style_feats: {style_feats.shape}"
                    )
                    # style_feats = self.blocks[0](style_feats)
                    style_feats = ResidualBlock(style_feats)
                    out = self.internal_adain(out, style_feats)

            if k == (i + 1):  # Do manifold mixup if k is greater 0
                out, targets = mixup_process(
                    out,
                    targets,
                    robust_samples,
                    self.num_classes,
                    mixup_alpha,
                    mixup_p,
                    cutmix_alpha,
                    cutmix_p,
                    generated_ratio,
                    manifold=True,
                    inplace=False,
                )
                out = noise_up(
                    out,
                    robust_samples=robust_samples,
                    add_noise_level=0.5,
                    mult_noise_level=0.5,
                    sparse_level=0.65,
                    l0_level=0.0,
                )

        return out, targets

    def _calc_mean_std(self, feat, eps=1e-5):
        """
        Calculate mean and standard deviation for each channel in the feature map (i.e., per image per channel or per instance).
        """

        N, C = feat.size()[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def internal_adain(self, content_feat, style_feat, epsilon=1e-5):
        """
        Perform Instance Normalization by normalizing content_feat and applying the style_feat's mean and std.
        """
        # Compute mean and std for content and style features
        content_mean, content_std = self._calc_mean_std(content_feat, epsilon)  # noqa: F821
        style_mean, style_std = self._calc_mean_std(style_feat, epsilon)

        # Normalize content feature and apply style moments
        normalized_content = (content_feat - content_mean) / content_std
        output = normalized_content * style_std + style_mean

        return output

    def pono(self, x, style_feats, epsilon=1e-5):
        """
        perform positional normalization with means and standard deviation along the channel dimension.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + epsilon).sqrt()

        style_mean = style_feats.mean(dim=1, keepdim=True)
        style_std = (style_feats.var(dim=1, keepdim=True) + epsilon).sqrt()
        normalized = (x - mean) / std

        stylized_feat = normalized * style_std + style_mean
        return stylized_feat
