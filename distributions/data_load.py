"""
Custom Dataloader configuration script
this script helps load out images, and preprocess them to be batched.
"""
import torch
from torch.utils.data import Dataset


import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_keypoints(image, keypoints):
    """
    Args:
        image: unprocessed images
        keypoints: x, y coordinates
    Returns: None
    """
    plt.scatter(keypoints[:, 0], keypoints[:, 1], marker='x', s=1, c='r')
    plt.imshow(np.squeeze(image), origin='lower')


class KeypointDataSet(Dataset):
    """
    Imports in Keypoints, and Image files.
    """

    def __init__(self, npy_file, root_dir, transform=None):

        self.data = np.load(npy_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data['b'])

    def __getitem__(self, idx):
        image = self.data['a'][idx]
        if image.shape[2] == 4:
            image = image[:, :, 0:3]
        key_pts = self.data['b'][idx]
        sample = {'image': image, 'keypoints': key_pts}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Normalize(object):
    """
    Preprocess/Normalize Keypoints, and Images.
    """
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy = image_copy/255.0

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        key_pts_copy = (key_pts_copy - 100)/50.0

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """
    This class Rescales to the desire scale.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))
        key_pts = key_pts * [new_w / w, new_h / h]
        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """
    This Class randomly crops images, and keypoints.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        key_pts = key_pts - [left, top]
        return {'image': image, 'keypoints': key_pts}


class ToTensor(object):
    """
    This class Transforms images, and keypoints to torch tensors.
    """
    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # if image has no grayscale color channel, add one
        if len(image.shape) == 2:
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image),
                'keypoints': torch.from_numpy(key_pts)}
