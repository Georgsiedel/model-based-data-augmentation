'''ResNeXt in PyTorch.

See the paper "Aggregated Residual Transformations for Deep Neural Networks" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.models import ct_model


class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1, activation_function=F.relu):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)
        self.activation_function = activation_function

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = self.activation_function(self.bn1(self.conv1(x)))
        out = self.activation_function(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation_function(out)
        return out


class ResNeXt(ct_model.CtModel):
    activation_function: object

    def __init__(self, num_blocks, cardinality, bottleneck_width, dataset, normalized, num_classes=10, factor=1, activation_function='relu'):
        super(ResNeXt, self).__init__(dataset=dataset, normalized=normalized, num_classes=num_classes)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.factor = factor
        self.activation_function = getattr(F, activation_function)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], factor, self.activation_function)
        self.layer2 = self._make_layer(num_blocks[1], 2, self.activation_function)
        self.layer3 = self._make_layer(num_blocks[2], 2, self.activation_function)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)
        self.blocks = [nn.Sequential(self.activation_function, self.conv1, self.bn1), self.layer1, self.layer2, self.layer3]

    def _make_layer(self, num_blocks, stride, activation_function):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride, activation_function))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x, targets=None, robust_samples=0, corruptions=None, mixup_alpha=0.0, mixup_p=0.0, manifold=False,
                manifold_noise_factor=1, cutmix_alpha=0.0, cutmix_p=0.0, noise_minibatchsize=1,
                concurrent_combinations=1, noise_sparsity=0.0, noise_patch_lower_scale = 1.0, noise_patch_upper_scale=1.0,
                generated_ratio=0.0):
        out = super(ResNeXt, self).forward_normalize(x)
        out, mixed_targets = super(ResNeXt, self).forward_noise_mixup(out, targets, robust_samples, corruptions,
                                        mixup_alpha, mixup_p, manifold, manifold_noise_factor, cutmix_alpha, cutmix_p,
                                        noise_minibatchsize, concurrent_combinations, noise_sparsity,
                                        noise_patch_lower_scale, noise_patch_upper_scale, generated_ratio)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        if self.training == True:
            return out, mixed_targets
        else:
            return out


def ResNeXt29_2x64d(dataset, normalized, num_classes, factor, activation_function='relu'):
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64, dataset=dataset, normalized=normalized,
                   num_classes=num_classes, factor=factor, activation_function=activation_function)

def ResNeXt29_4x64d(dataset, normalized, num_classes, factor, activation_function='relu'):
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64, dataset=dataset, normalized=normalized,
                   num_classes=num_classes, factor=factor, activation_function=activation_function)

def ResNeXt29_8x64d(dataset, normalized, num_classes, factor, activation_function='relu'):
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64, dataset=dataset, normalized=normalized,
                   num_classes=num_classes, factor=factor, activation_function=activation_function)

def ResNeXt29_32x4d(dataset, normalized, num_classes, factor, activation_function='relu'):
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4, dataset=dataset, normalized=normalized,
                   num_classes=num_classes, factor=factor, activation_function=activation_function)

def test_resnext():
    net = ResNeXt29_2x64d()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test_resnext()
