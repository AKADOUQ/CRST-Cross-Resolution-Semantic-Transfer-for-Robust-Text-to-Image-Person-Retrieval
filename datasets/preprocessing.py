import random
import math


class RandomErasing(object):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

import torch
import torch.nn.functional as F

class RandomDownsample(object):

    def __init__(self, levels=((384, 128), (128, 64), (64, 32), (32, 16)),
                 probs=None, mode='bicubic'):
        self.levels = list(levels)
        self.probs = probs
        self.mode = mode

    def _interpolate(self, x, size):
        kwargs = dict(size=size, mode=self.mode)
        if self.mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
            kwargs['align_corners'] = False
        try:
            return F.interpolate(x, antialias=True, **kwargs)
        except TypeError:
            return F.interpolate(x, **kwargs)

    def __call__(self, img_tensor):
        # img_tensor: [C, H, W]
        device = img_tensor.device
        if self.probs is None:
            res_label = int(torch.randint(0, len(self.levels), (1,), device=device).item())
        else:
            probs = torch.tensor(self.probs, dtype=torch.float, device=device)
            probs = probs / probs.sum()
            res_label = int(torch.multinomial(probs, 1).item())

        if res_label == 0:
            return img_tensor.clone(), 0

        target_h, target_w = self.levels[res_label]
        orig_h, orig_w = img_tensor.shape[-2:]
        x = img_tensor.unsqueeze(0)
        x_small = self._interpolate(x, size=(target_h, target_w))
        x_restore = self._interpolate(x_small, size=(orig_h, orig_w))
        return x_restore.squeeze(0), res_label
