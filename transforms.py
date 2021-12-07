import random
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class Crop(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, image, target):
        pass


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


def Resize(h, w, shape, interp=Image.BILINEAR):



    return ResizeTransform(
        h, w, shape[0][1], shape[0][2], interp
    )


class ResizeTransform(object):
    """
        Resize the image to a target size.
        """

    def __init__(self, h, w, new_h, new_w, interp):
        """
        Args:
            h, w (int): original image size
            new_h, new_w (int): new image size
            interp: PIL interpolation methods
        """
        # TODO decide on PIL vs opencv
        super().__init__()
        self.h = h
        self.w = w
        self.new_h = new_h
        self.new_w = new_w
        self.interp = interp

    def apply_image(self, img, interp=None):
        assert img.shape[1:] == (self.h, self.w)
        img = np.asarray(img.cpu()).transpose(1, 2, 0)
        pil_image_0 = Image.fromarray(img[:, :, 0])
        pil_image_1 = Image.fromarray(img[:, :, 1])
        pil_image_2 = Image.fromarray(img[:, :, 2])

        interp_method = interp if interp is not None else self.interp
        pil_image_0 = pil_image_0.resize((self.new_w, self.new_h), interp_method)
        pil_image_1 = pil_image_1.resize((self.new_w, self.new_h), interp_method)
        pil_image_2 = pil_image_2.resize((self.new_w, self.new_h), interp_method)
        pil_image_0 = np.asarray(pil_image_0)
        pil_image_1 = np.asarray(pil_image_1)
        pil_image_2 = np.asarray(pil_image_2)
        pil_image = np.concatenate((np.expand_dims(pil_image_0, 2), np.expand_dims(pil_image_1, 2), np.expand_dims(pil_image_2, 2)), axis=2).transpose(2, 0, 1)

        return pil_image

    def apply_coords(self, coords):
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / self.w)
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / self.h)
        return coords


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target
