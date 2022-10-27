from abc import ABC
from torch.utils.data import Dataset
import numpy as np
import torchio as tio
import os.path as op
from pathlib import Path
import sys


class SpinalCordDataset(Dataset, ABC):
    """
    Create own dataset for usage in PyTorch. Enables application of image transformations.
    """
    def __init__(self, annotations_files, img_files, transform=None, target_transform=None):
        self.targets = annotations_files
        self.img_files = img_files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        image = img_file.data
        label = self.targets[idx].data
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def divide_data(image_dimensions=(160, 64, 35)):
    """
    Splits the data into subset for training and validation and a subset for testing the model.
    :param image_dimensions: Desired dimension of images. Images are cropped/padded accordingly
    :return: Tuple of (train_val_set, test_set)
    """
    # list of images and labels
    image_files = []
    image_labels = []
    test_images = []
    test_labels = []

    # Paths of the data
    experiment1 = Path(op.join(sys.path[0], '../dataset/Experiment1'))
    experiment2 = Path(op.join(sys.path[0], '../dataset/Experiment2'))
    experiment3 = Path(op.join(sys.path[0], '../dataset/Experiment3'))

    # experiment 1 has the least number of participants, randomize indices to pick from each experiment two subjects for
    # testing
    exp1_files = 12
    np.random.seed(42)
    indices = list(range(exp1_files))
    np.random.shuffle(indices)

    index = 0
    for file_path in experiment1.iterdir():
        for image in file_path.rglob('*seg.nii.gz'):
            if index in indices[:2]:
                test_images.append(tio.ScalarImage(image))
            else:
                image_files.append(tio.ScalarImage(image))
        for label in file_path.rglob('*sc.nii.gz'):
            if index in indices[:2]:
                test_labels.append(tio.ScalarImage(label))
            else:
                image_labels.append(tio.ScalarImage(label))
        index += 1

    index = 0
    for file_path in experiment2.iterdir():
        for image in file_path.rglob('*mean.nii.gz'):
            if index in indices[:2]:
                test_images.append(tio.ScalarImage(image))
            else:
                image_files.append(tio.ScalarImage(image))
        for label in file_path.rglob('*sc.nii.gz'):
            if index in indices[:2]:
                test_labels.append(tio.ScalarImage(label))
            else:
                image_labels.append(tio.ScalarImage(label))
        index += 1

    index = 0
    for file_path in experiment3.iterdir():
        for image in file_path.rglob('*seg.nii.gz'):
            if index in indices[:2]:
                test_images.append(tio.ScalarImage(image))
            else:
                image_files.append(tio.ScalarImage(image))
        for label in file_path.rglob('*sc.nii.gz'):
            if index in indices[:2]:
                test_labels.append(tio.ScalarImage(label))
            else:
                image_labels.append(tio.ScalarImage(label))
        index += 1

    # Create Datasets
    dataset = SpinalCordDataset(image_labels, image_files, tio.CropOrPad(image_dimensions),
                                tio.CropOrPad(image_dimensions))
    test_set = SpinalCordDataset(test_labels, test_images, tio.CropOrPad(image_dimensions),
                                 tio.CropOrPad(image_dimensions))

    return dataset, test_set
