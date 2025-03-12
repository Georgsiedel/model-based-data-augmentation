# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#This source code is taken from https://github.com/facebookresearch/augmentation-corruption

import abc
import numpy as np
from scipy.fftpack import ifft2
from scipy.ndimage import gaussian_filter, rotate, zoom
from skimage.draw import line_aa

transform_list = [
    SingleFrequencyGreyscale,
    CocentricSineWaves,
    PlasmaNoise,
    CausticNoise,
    PerlinNoise,
    BlueNoise,
    BrownishNoise,
    TransverseChromaticAbberation,
    CircularMotionBlur,
    CheckerBoardCutOut,
    Sparkles,
    InverseSparkles,
    Lines,
    BlueNoiseSample,
    PinchAndTwirl,
    CausticRefraction,
    Ripple
]

transform_dict = {t.name : t for t in transform_list}

def build_transform(name, severity, dataset):
    assert dataset in ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet'],\
            "Only cifar and imagenet image resolutions are supported."
    if dataset in ['CIFAR10', 'CIFAR100']: 
        im_size = 32
    elif dataset in ['TinyImageNet']: 
        im_size = 64
    else:
        im_size = 224
    return transform_dict[name](severity=severity, im_size=im_size)


class Transform(abc.ABC):

    name = "abstract_transform"

    def __init__(self, severity, im_size, record=False, max_intensity=False, **kwargs):
        self.im_size = im_size
        self.severity = severity
        self.record = record
        self.max_intensity = max_intensity

    @abc.abstractmethod
    def transform(self, image, **kwargs):
        ...

    @abc.abstractmethod
    def sample_parameters(self):
        ...

    def __call__(self, image):
        params = self.sample_parameters()
        out = self.transform(image, **params)
        if self.record:
            return out, params
        return out


    def convert_to_numpy(self, params):
        out = []
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                out.extend(v.flatten().tolist())
            elif is_iterable(v):
                out.extend([x for x in v])
            else:
                out.append(v)
        return np.array(out)

    def convert_from_numpy(self, numpy_record):
        param_signature = self.sample_parameters()
        #assert len(param_signature.keys())<=len(numpy_record), "Mismatched numpy_record."
        offset = 0
        for k, v in param_signature.items():
            if isinstance(v, np.ndarray):
                num = len(v.flatten())
                data = numpy_record[offset:offset+num]
                if v.dtype==np.int or v.dtype==np.uint:
                    data = np.round(data, 3)
                data = data.astype(v.dtype)
                param_signature[k] = data.reshape(v.shape)
                offset += num
            elif is_iterable(v):
                data = []
                for x in v:
                    if type(x) == 'int':
                        data.append(int(np.round(numpy_record[offset],3)))
                    else:
                        data.append(type(x)(numpy_record[offset]))
                    offset += 1
                param_signature[k] = data
            else:
                if type(v) == 'int':
                    param_signature[k] = int(np.round(numpy_record[offset],3))
                else:
                    param_signature[k] = type(v)(numpy_record[offset])
                offset += 1
        return param_signature

def smoothstep(low, high, x):
    x = np.clip(x, low, high)
    x = (x - low) / (high - low)
    return np.clip(3 * (x ** 2) - 2 * (x ** 3), 0, 1)


def bilinear_interpolation(image, point):
    l = int(np.floor(point[0]))
    u = int(np.floor(point[1]))
    r, d = l+1, u+1
    lu = image[l,u,:] if l >= 0 and l < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    ld = image[l,d,:] if l >= 0 and l < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    ru = image[r,u,:] if r >= 0 and r < image.shape[0]\
            and u >= 0 and u < image.shape[1] else np.array([0,0,0])
    rd = image[r,d,:] if r >= 0 and r < image.shape[0]\
            and d >= 0 and d < image.shape[1] else np.array([0,0,0])
    al = lu * (1.0 - point[1] + u) + ld * (1.0 - d + point[1])
    ar = ru * (1.0 - point[1] + u) + rd * (1.0 - d + point[1])
    out = al * (1.0 - point[0] + l) + ar * (1.0 - r + point[0])
    return out

def int_parameter(level, maxval):
  return int(level * maxval / 10)

def float_parameter(level, maxval):
  return float(level) * maxval / 10.

class PerlinNoiseGenerator(object):
    def __init__(self, random_state=None):
        self.rand = np.random if random_state is None else random_state

        B = 256
        N = 16*256

        def normalize(arr):
            return arr / np.linalg.norm(arr)

        self.p = np.arange(2*B+2)
        self.g = np.array([normalize((random_state.randint(low=0, high=2**31, size=2) % (2*B) - B )/ B)\
                for i in range(2*B+2)])


        for i in np.arange(B-1,-1,-1):
            k = self.p[i]
            j = self.rand.randint(low=0, high=2**31) % B
            self.p[i] = self.p[j]
            self.p[j] = k

        for i in range(B+2):
            self.p[B+i] = self.p[i]
            self.g[B+i,:] = self.g[i,:]
        self.B = B
        self.N = N


    def s_curve(t):
        return t**2 * (3.0 - 2.0 * t)

    def noise(self, x, y):

        t = x + self.N
        bx0 = int(t) % self.B
        bx1 = (bx0+1) % self.B
        rx0 = t % 1
        rx1 = rx0 - 1.0

        t = y + self.N
        by0 = int(t) % self.B
        by1 = (by0+1) % self.B
        ry0 = t % 1
        ry1 = ry0 - 1.0

        i = self.p[bx0]
        j = self.p[bx1]

        b00 = self.p[i + by0]
        b10 = self.p[j + by0]
        b01 = self.p[i + by1]
        b11 = self.p[j + by1]

        sx = PerlinNoiseGenerator.s_curve(rx0)
        sy = PerlinNoiseGenerator.s_curve(ry0)

        u = rx0 * self.g[b00,0] + ry0 * self.g[b00,1]
        v = rx1 * self.g[b10,0] + ry0 * self.g[b10,1]
        a = u + sx * (v - u)

        u = rx0 * self.g[b01,0] + ry1 * self.g[b01,1]
        v = rx1 * self.g[b11,0] + ry1 * self.g[b11,1]
        b = u + sx * (v - u)

        return 1.5 * (a + sy * (b - a))

    def turbulence(self, x, y, octaves):
        t = 0.0
        f = 1.0
        while f <= octaves:
            t += np.abs(self.noise(f*x, f*y)) / f
            f = f * 2
        return t

class SingleFrequencyGreyscale(Transform):

    name = 'single_frequency_greyscale'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        freq_mag = np.random.uniform(low=-np.pi, high=np.pi)
        freq_2 = np.random.uniform(low=-abs(freq_mag), high=abs(freq_mag))
        freq = np.array([freq_mag, freq_2])[np.random.permutation(2)]
        phase = np.random.uniform(low=0, high=2*np.pi)
        intensity = float_parameter(self.severity, 196)
        return { 'freq' : freq, 'phase' : phase, 'intensity' : intensity}

    def transform(self, image, freq, phase, intensity):
        noise = np.array([[np.sin(x * freq[0] + y * freq[1] + phase)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((noise, noise, noise), axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['freq'].tolist() + [params['phase'], params['intensity']])

    def convert_from_numpy(self, numpy_record):
        return {'freq' : numpy_record[0:2],
                'phase' : numpy_record[2],
                'intensity' : numpy_record[3]
                }

class CocentricSineWaves(Transform):

    name = 'cocentric_sine_waves'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        offset = np.random.uniform(low=0, high=self.im_size, size=2)
        freq = np.random.uniform(low=0, high=10)
        amplitude = np.random.uniform(low=0, high=self.im_size/10)
        ring_width = np.random.uniform(low=0, high=self.im_size/10)
        intensity = [float_parameter(self.severity, 128) for i in range(3)]

        return { 'offset' : offset,
                 'freq' : freq,
                 'amplitude' : amplitude,
                 'ring_width' : ring_width,
                 'intensity' : intensity
                }

    def transform(self, image, offset, freq, amplitude, ring_width, intensity):

        def calc_intensity(x, y, x0, y0, freq, amplitude, ring_width):
            angle = np.arctan2(x-x0, y-y0) * freq
            distance = ((np.sqrt((x-x0)**2 + (y-y0)**2) + np.sin(angle) * amplitude) % ring_width) / ring_width
            distance -= 1/2
            return distance

        noise = np.array([[calc_intensity(x, y, offset[0], offset[1], freq, amplitude, ring_width)\
                    for x in range(self.im_size)] for y in range(self.im_size)])
        noise = np.stack((intensity[0] * noise, intensity[1] * noise, intensity[2] * noise), axis=2)

        return np.clip(image + noise, 0, 255).astype(np.uint8)

    def convert_to_numpy(self, params):
        return np.array(params['offset'].tolist() + [params['freq'], params['amplitude'], params['ring_width']] + params['intensity'])

    def convert_from_numpy(self, numpy_record):
        return {'offset' : numpy_record[0:2].tolist(),
                'freq' : numpy_record[2],
                'amplitude' : numpy_record[3],
                'ring_width' : numpy_record[4],
                'intensity' : numpy_record[4:7].tolist()
                }
        

class PlasmaNoise(Transform):

    name = 'plasma_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.0, high=6*np.pi)
        iterations = np.random.randint(low=4, high=7)
        sharpness = np.random.uniform(low=0.5, high=1.0)
        scale = np.random.uniform(low=0.075, high=0.2) * self.im_size
        intensity = float_parameter(self.severity,64)
        return {'time' : time, 'iterations' : iterations, 'sharpness' : sharpness,
                'scale' : scale, 'intensity' : intensity}

    def transform(self, image, time, iterations, sharpness, scale, intensity):

        def kernel(x, y, rand, iters, sharp, scale):
            x /= scale
            y /= scale
            i = np.array([1.0, 1.0, 1.0, 0.0])
            for s in range(iters):
                r = np.array([np.cos(y * i[0] - i[3] + rand / i[1]), np.sin(x * i[0] - i[3] + rand / i[1])]) / i[2]
                r += np.array([-r[1],r[0]]) * 0.3
                x += r[0]
                y += r[1]
                i *= np.array([1.93, 1.15, (2.25 - sharp), rand * i[1]])
            r = np.sin(x - rand)
            b = np.sin(y + rand)
            g = np.sin((x + y + np.sin(rand))*0.5)
            return [r,g,b]


        noise = np.array([[kernel(x,y, time, iterations, sharpness, scale)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip((1-intensity/255) * image + intensity * noise, 0, 255).astype(np.uint8)

class CausticNoise(Transform):

    name = 'caustic_noise'
    tags = ['new_corruption']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        intensity = float_parameter(self.severity, 255)

        return { 'time' : time, 'size' : size, 'intensity' : intensity}

    def transform(self, image, time, size, intensity):

        def kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])

        noise = np.array([[kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(image + intensity  *  noise, 0, 255).astype(np.uint8)

class Sparkles(Transform):

    name = 'sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        centers = np.random.uniform(low=0, high=self.im_size, size=(5, 2))
        radii = np.array([float_parameter(self.severity, 0.1)\
                for i in range(5)]) * self.im_size
        amounts = np.array([50 for i in range(5)])
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        nrays = np.random.randint(low=50, high=200, size=5)

        return {'centers' : centers, 'radii' : radii, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'nrays' : nrays, 'amounts' : amounts
                }

    def transform(self, image, centers, radii, nrays, amounts, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return value + f * (color - value)

        random_state = np.random.RandomState(seed=seed)
        for center, rays, amount, radius in zip(centers, nrays, amounts, radii):
            ray_lengths = [max(1,radius + randomness / 100.0 * radius * random_state.randn())\
                for i in range(rays)]

            image = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8)


class InverseSparkles(Transform):

    name = 'inverse_sparkles'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        center = np.random.uniform(low=0.25, high=0.75, size=2) * self.im_size
        radius = 0.25 * self.im_size
        amount = 100
        amount = float_parameter(self.severity, 65)
        amount = 100 - amount
        color = np.array([255, 255, 255])
        randomness = 25
        seed = np.random.randint(low=0, high=2**32)
        rays = np.random.randint(low=50, high=200)

        return {'center' : center, 'radius' : radius, 'color' : color, 'randomness' : randomness,
                'seed' : seed, 'rays' : rays, 'amount' : amount
                }

    def transform(self, image, center, radius, rays, amount, color, randomness, seed):

        def kernel(point, value, center, radius, ray_lengths, amount, color):
            rays = len(ray_lengths)
            dp = point - center
            dist = np.linalg.norm(dp)
            angle = np.arctan2(dp[1], dp[0])
            d = (angle + np.pi) / (2 * np.pi) * rays
            i = int(d)
            f = d - i 

            if radius != 0:
                length = ray_lengths[i % rays] + f * (ray_lengths[(i+1) % rays] - ray_lengths[i % rays])
                g = length**2 / (dist**2 + 1e-4)
                g = g ** ((100 - amount) / 50.0)
                f -= 0.5
                f = 1 - f**2
                f *= g
            f = np.clip(f, 0, 1)
            return color + f * (value - color)

        random_state = np.random.RandomState(seed=seed)
        ray_lengths = [radius + randomness / 100.0 * radius * random_state.randn()\
                for i in range(rays)]

        out = np.array([[kernel(np.array([y,x]), image[y,x,:].astype(np.float32), center, radius, ray_lengths, amount, color)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(out, 0, 255).astype(np.uint8)

class PerlinNoise(Transform):

    name = 'perlin_noise'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        m = np.array([[1,0],[0,1]]) / (32 * self.im_size / 224)
        turbulence = 16.0
        gain = 0.5
        bias = 0.5
        alpha = float_parameter(self.severity, 0.50)
        seed = np.random.randint(low=0, high=2**32)
        return {'m': m, 'turbulence' : turbulence, 'seed': seed,
                'gain': gain, 'bias': bias, 'alpha': alpha}

    def transform(self, image, m, turbulence, seed, gain, bias, alpha):
        
        random_state = np.random.RandomState(seed=seed)
        noise = PerlinNoiseGenerator(random_state)

        def kernel(point, m, turbulence, gain, bias):
            npoint = np.matmul(point, m)
            f = noise.turbulence(npoint[0], npoint[1], turbulence)\
                    if turbulence != 1.0 else noise.noise(npoint[0], npoint[1])
            f = gain * f + bias
            return np.clip(np.array([f,f,f]),0,1.0)

        noise = np.array([[kernel(np.array([y,x]),m,turbulence,gain, bias) for x in range(self.im_size)]\
                for y in range(self.im_size)])
        out = (1 - alpha) * image.astype(np.float32) + 255 * alpha * noise
        return np.clip(out, 0, 255).astype(np.uint8)

class BlueNoise(Transform):

    name = 'blue_noise'
    tags = ['new_corruption']


    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(self.severity, 196)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)

class BrownishNoise(Transform):

    name = 'brownish_noise'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        intensity = float_parameter(self.severity, 64)

        return {'seed' : seed, 'intensity' : intensity}

    def gen_noise(self, random_state):
        center = self.im_size / 2
        power = np.array([[1/(np.linalg.norm(np.array([x,y])-center)**2+1)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        return noise

    def transform(self, image, seed, intensity):
        random_state = np.random.RandomState(seed=seed)
        noise = np.stack([self.gen_noise(random_state) for i in range(3)],axis=2)

        return np.clip(image + intensity * noise, 0, 255).astype(np.uint8)
 
class CheckerBoardCutOut(Transform):

    name = 'checkerboard_cutout'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        angle = np.random.uniform(low=0, high=2*np.pi)
        scales = np.maximum(np.random.uniform(low=0.1, high=0.25) * self.im_size, 1)
        scales = (scales, scales)
        fraction = float_parameter(self.severity, 1.0)
        seed = np.random.randint(low=0, high=2**32)

        return {'angle' : angle, 'scales' : scales, 'fraction' : fraction, 'seed' : seed}

    def transform(self, image, scales, angle, fraction, seed):
        random_state = np.random.RandomState(seed=seed)
        grid = random_state.uniform(size=(int(4*self.im_size//scales[0]), int(4*self.im_size//scales[1]))) < fraction
        
        def mask_kernel(point, scales, angle, grid):
            nx = (np.cos(angle) * point[0] + np.sin(angle) * point[1]) / scales[0]
            ny = (-np.sin(angle) * point[0] + np.cos(angle) * point[1]) / scales[1]
            return (int(nx % 2) != int(ny % 2)) or not grid[int(nx),int(ny)]

        out = np.array([[image[y,x,:] if mask_kernel([y,x], scales, angle, grid) else np.array([128,128,128])\
                for x in range(self.im_size)] for y in range(self.im_size)])
        return np.clip(out, 0, 255).astype(np.uint8)


class Lines(Transform):

    name = 'lines'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        length = 1.0
        density = float_parameter(self.severity, 1.0)
        angle = np.random.uniform(low=0.0, high=2*np.pi)
        angle_variation = np.random.uniform(low=0.1, high=1.0)
        seed = np.random.randint(low=0, high=2**32)

        return {'length' : length, 'density' : density, 'angle' : angle, 'angle_variation' : angle_variation, 'seed' : seed}

    def transform(self, image, length, density, angle, angle_variation, seed):

        num_lines = int(density * self.im_size)
        l = length * self.im_size
        random_state = np.random.RandomState(seed=seed)
        out = image.copy()
        for i in range(num_lines):
            x = self.im_size * random_state.uniform()
            y = self.im_size * random_state.uniform()
            a = angle + 2 * np.pi * angle_variation * (random_state.uniform() - 0.5)
            s = np.sin(a) * l
            c = np.cos(a) * l
            x1 = int(x-c)
            x2 = int(x+c)
            y1 = int(y-s)
            y2 = int(y+s)
            rxc, ryc, rval = line_aa(x1, y1, x2, y2)
            xc, yc, val = [], [], []
            for rx, ry, rv in zip(rxc, ryc, rval):
                if rx >= 0 and ry >= 0 and rx < self.im_size and ry < self.im_size:
                    xc.append(rx)
                    yc.append(ry)
                    val.append(rv)
            xc, yc, val = np.array(xc, dtype=np.int), np.array(yc, dtype=np.int), np.array(val)
            out[xc, yc, :] = (1.0 - val.reshape(-1,1)) * out[xc, yc, :].astype(np.float32) + val.reshape(-1,1)*128
        return out.astype(np.uint8)


class BlueNoiseSample(Transform):

    name = 'blue_noise_sample'
    tags = ['new_corruption', 'imagenet_c_bar', 'cifar_c_bar']

    def sample_parameters(self):
        seed = np.random.randint(low=0, high=2**32)
        threshold = float_parameter(self.severity, 3.0) - 2.5

        return {'seed' : seed, 'threshold' : threshold}

    def transform(self, image, seed, threshold):
        random_state = np.random.RandomState(seed=seed)

        center = self.im_size / 2
        power = np.array([[np.linalg.norm(np.array([x,y])-center)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        phases = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size, self.im_size//2))
        if self.im_size % 2 == 0:
            phases = np.concatenate((phases, phases[::-1,::-1]), axis=1)
        else:
            center_freq = random_state.uniform(low=0, high=2*np.pi, size=(self.im_size//2, 1))
            center_freq = np.concatenate((center_freq, np.array([[0.0]]), center_freq[::-1,:]), axis=0)
            phases = np.concatenate((phases, center_freq, phases[::-1,::-1]), axis=1)
        fourier_space_noise = power * (np.cos(phases) + np.sin(phases) * 1j)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=0)
        fourier_space_noise = np.roll(fourier_space_noise, self.im_size//2, axis=1)


        noise = np.real(ifft2(fourier_space_noise))
        noise = noise / np.std(noise)
        mask = noise > threshold
        out = image * mask.reshape(self.im_size, self.im_size, 1)


        return np.clip(out, 0, 255).astype(np.uint8)

class CausticRefraction(Transform):

    name = 'caustic_refraction'
    tags = ['new_corruption', 'imagenet_c_bar']

    def sample_parameters(self):
        time = np.random.uniform(low=0.5, high=2.0)
        size = np.random.uniform(low=0.75, high=1.25) * self.im_size
        #size = self.im_size
        eta = 4.0
        lens_scale = float_parameter(self.severity, 0.5*self.im_size)
        lighting_amount = float_parameter(self.severity, 2.0)
        softening = 1

        return { 'time' : time, 'size' : size, 'eta' : eta, 'lens_scale' : lens_scale, 'lighting_amount': lighting_amount, 'softening' : softening}

    def transform(self, image, time, size, eta, lens_scale, lighting_amount, softening):

        def caustic_noise_kernel(point, time, size):
            point = point / size
            p = (point % 1) * 6.28318530718 - 250

            i = p.copy()
            c = 1.0
            inten = 0.005

            for n in range(5):
                t = time * (1.0 - (3.5 / (n+1)))
                i = p + np.array([np.cos(t-i[0])+np.sin(t+i[1]),np.sin(t-i[1])+np.cos(t+i[0])])
                length = np.sqrt((p[0] / (np.sin(i[0]+t)/inten))**2 + (p[1] / (np.cos(i[1]+t)/inten))**2)
                c += 1.0/length

            c /= 5.0
            c = 1.17 - c ** 1.4
            color = np.clip(np.abs(c) ** 8.0, 0, 1) 
            return np.array([color, color, color])


        def refract(incident, normal, eta):
            if np.abs(np.dot(incident, normal)) >= 1.0 - 1e-3:
                return incident
            angle = np.arccos(np.dot(incident, normal))
            out_angle = np.arcsin(np.sin(angle) / eta)
            out_unrotated = np.array([np.cos(out_angle), np.sin(out_angle), 0.0])
            spectator_dim = np.cross(incident, normal)
            spectator_dim /= np.linalg.norm(spectator_dim)
            orthogonal_dim = np.cross(normal, spectator_dim)
            rotation_matrix = np.stack((normal, orthogonal_dim, spectator_dim), axis=0)
            return np.matmul(np.linalg.inv(rotation_matrix), out_unrotated)

        def luma_at_offset(image, origin, offset):
            pixel_value = image[origin[0]+offset[0], origin[1]+offset[1], :]\
                    if origin[0]+offset[0] >= 0 and origin[0]+offset[0] < image.shape[0]\
                    and origin[1]+offset[1] >= 0 and origin[1]+offset[1] < image.shape[1]\
                    else np.array([0.0,0.0,0])
            return np.dot(pixel_value, np.array([0.2126, 0.7152, 0.0722]))

        def luma_based_refract(point, image, caustics, eta, lens_scale, lighting_amount):
            north_luma = luma_at_offset(caustics, point, np.array([0,-1]))
            south_luma = luma_at_offset(caustics, point, np.array([0, 1]))
            west_luma = luma_at_offset(caustics, point, np.array([-1, 0]))
            east_luma = luma_at_offset(caustics, point, np.array([1,0]))

            lens_normal = np.array([east_luma - west_luma, south_luma - north_luma, 1.0])
            lens_normal = lens_normal / np.linalg.norm(lens_normal)

            refract_vector = refract(np.array([0.0, 0.0, 1.0]), lens_normal, eta) * lens_scale
            refract_vector = np.round(refract_vector, 3)

            out_pixel = bilinear_interpolation(image, point+refract_vector[0:2])
            out_pixel += (north_luma - south_luma) * lighting_amount
            out_pixel += (east_luma - west_luma) * lighting_amount

            return np.clip(out_pixel, 0, 1)

        noise = np.array([[caustic_noise_kernel(np.array([y,x]), time, size)\
                for x in range(self.im_size)] for y in range(self.im_size)])
        noise = gaussian_filter(noise, sigma=softening)

        image = image.astype(np.float32) / 255
        out = np.array([[luma_based_refract(np.array([y,x]), image, noise, eta, lens_scale, lighting_amount)\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip((out * 255).astype(np.uint8), 0, 255)

class PinchAndTwirl(Transform):

    name = 'pinch_and_twirl'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        num_per_axis = 5 if self.im_size==224 else 3
        angles = np.array([np.random.choice([1,-1]) * float_parameter(self.severity, np.pi/2) for i in range(num_per_axis ** 2)]).reshape(num_per_axis, num_per_axis)

        amount = float_parameter(self.severity, 0.4) + 0.1
        return {'num_per_axis' : num_per_axis, 'angles' : angles, 'amount' : amount}

    def transform(self, image, num_per_axis, angles, amount):

        def warp_kernel(point, center, radius, amount, angle):
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            dist = np.linalg.norm(point - center)

            if dist > radius or np.round(dist, 3) == 0.0:
                return point

            d = dist / radius
            t = np.sin(np.pi * 0.5 * d) ** (- amount)

            dx *= t
            dy *= t

            e = 1 - d
            a = angle * (e ** 2)
            
            out = center + np.array([dx*np.cos(a) - dy*np.sin(a), dx*np.sin(a) + dy*np.cos(a)])

            return out

        out = image.copy().astype(np.float32)
        grid_size = self.im_size // num_per_axis
        radius = grid_size / 2
        for i in range(num_per_axis):
            for j in range(num_per_axis):
                l, r = i * grid_size, (i+1) * grid_size
                u, d = j * grid_size, (j+1) * grid_size
                center = np.array([u+radius, l+radius])
                out[u:d,l:r,:] = np.array([[bilinear_interpolation(out, warp_kernel(np.array([y,x]), center, radius, amount, angles[i,j]))\
                        for x in np.arange(l,r)] for y in np.arange(u,d)])

        return np.clip(out, 0, 255).astype(np.uint8)

class Ripple(Transform):

    name = 'ripple'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amplitudes = np.array([float_parameter(self.severity, 0.025)\
                for i in range(2)]) * self.im_size
        wavelengths = np.random.uniform(low=0.1, high=0.3, size=2) * self.im_size
        phases = np.random.uniform(low=0, high=2*np.pi, size=2)
        return {'amplitudes' : amplitudes, 'wavelengths' : wavelengths, 'phases' : phases}

    def transform(self, image, wavelengths, phases, amplitudes):

        def warp_kernel(point, wavelengths, phases, amplitudes):
            return point + amplitudes * np.sin(2 * np.pi * point / wavelengths + phases)

        image = np.array([[bilinear_interpolation(image, warp_kernel(np.array([y,x]), wavelengths, phases, amplitudes))\
                for x in range(self.im_size)] for y in range(self.im_size)])

        return np.clip(image, 0, 255).astype(np.uint8) 


class TransverseChromaticAbberation(Transform):

    name = 'transverse_chromatic_abberation'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        scales = np.array([float_parameter(self.severity, 0.5)\
                for i in range(3)])
        scale = float_parameter(self.severity, 0.5)
        scales = np.array([1.0, 1.0+scale/2, 1.0+scale])
        scales = scales[np.random.permutation(3)]

        return { 'scales' : scales }

    def transform(self, image, scales):
        out = image.copy()
        for c in range(3):
            zoomed = zoom(image[:,:,c], scales[c], prefilter=False)
            edge = (zoomed.shape[0]-self.im_size)//2
            out[:,:,c] = zoomed[edge:edge+self.im_size, edge:edge+self.im_size]
        return out.astype(np.uint8)
            
    def convert_to_numpy(self, params):
        return params['scales'].flatten()

    def convert_from_numpy(self, numpy_record):
        return {'scales' : numpy_record}


class CircularMotionBlur(Transform):

    name = 'circular_motion_blur'
    tags = ['new_corruption', 'cifar_c_bar']

    def sample_parameters(self):
        amount = float_parameter(self.severity,15)

        return {'amount' : amount}

    def transform(self, image, amount):

        num = 21
        factors = []
        rotated = []
        image = image.astype(np.float32) / 255
        for i in range(num):
            angle = (2*i/(num-1) - 1) * amount
            rotated.append(rotate(image, angle, reshape=False))
            factors.append(np.exp(- 2*(2*i/(num-1)-1)**2))
        out = np.zeros_like(image)
        for i, f in zip(rotated, factors):
            out += f * i
        out /= sum(factors)
        return np.clip(out*255, 0, 255).astype(np.uint8)
    