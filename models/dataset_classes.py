from abc import ABC
from torch.utils.data import Dataset, Subset
import numpy as np
import torchio as tio
import os.path as op
from pathlib import Path
import sys
import re


class SpinalCordDataset(Dataset, ABC):
    """
    Create own dataset for usage in PyTorch. Enables application of image transformations.

    :ivar targets: labels / ground truth masks
    :ivar img_files: fMRI spinal cord images
    :ivar transform: transformations to be applied on the fMRI images
    :ivar target_transform: transformations to be applied on the targets
    """
    def __init__(self, annotations_files, img_files, transform=None, target_transform=None):
        self.targets = annotations_files
        self.img_files = img_files
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx, new_shape=(1, 160, -1)):
        img_file = self.img_files[idx]
        image = img_file.data
        label = self.targets[idx].data
        if self.transform:
            image = self.transform(image).reshape(new_shape)
        if self.target_transform:
            label = self.target_transform(label).reshape(new_shape)
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
    experiment1 = Path('dataset/Experiment1')
    experiment2 = Path('dataset/Experiment2')
    experiment3 = Path('dataset/Experiment3')

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


class Experiment4(ABC):
    """
    Creates a dataset consisting of the fMRI images of one subject gained by splitting the fMRI sequence. Images for
    which the mean mask cannot be applied have to be provided with a new mask.

    :ivar subjects: list of torchio Subject instances
    """
    def __init__(self, subject_path, new_labels):
        """
        :param subject_path: relative path to the fMRI sequence
        :param new_labels: List of sequence numbers which have a different mask
        """
        single_imgs = Path(op.join(sys.path[0], subject_path + '/single_images'))
        regex = re.compile(r'\d+')
        cop = tio.CropOrPad((160, 64, 40))

        self.subjects = []
        # assign masks to images
        for file in single_imgs.iterdir():
            idx = int(regex.findall(str(file))[2])
            if idx in new_labels:
                label_path = op.join(sys.path[0], subject_path + f'/mask_sc_vol{idx}.nii.gz')
            else:
                label_path = op.join(sys.path[0], subject_path + '/mask_sc.nii.gz')

            # create torchio subject instance and pad image and mask for later augmentation
            self.subjects.append(tio.Subject(img=cop(tio.ScalarImage(file)), label=cop(tio.LabelMap(label_path))))

    def create_dataset(self, augmentation=True):
        """
        Applies data augmentation to the images and labels. The dataset instance from torchio is used to get the
        inverse transformation.

        :return: dataset
        """
        lateral = tio.transforms.RandomAffine(scales=(1, 1), degrees=(-3, 3), translation=0, center='image')
        affine = tio.transforms.RandomAffine(scales=(1, 1), degrees=(-3, 3),
                                             translation=(1.5, 2.5, 0), center='image')
        scaling = tio.transforms.RandomAffine(scales=(1, 1.5), degrees=(0, 0), center='image', isotropic=True)
        elastic = tio.transforms.RandomElasticDeformation(num_control_points=(5, 10, 10),
                                                          max_displacement=(1.5, 4, 4.5),
                                                          locked_borders=2,
                                                          image_interpolation='linear',
                                                          label_interpolation='nearest')
        none = tio.transforms.RandomAffine(scales=(0, 0), degrees=(0, 0), translation=0, center='image',
                                           include='img')

        # create combinations of transformations
        trans1 = tio.transforms.Compose([lateral, affine])
        trans2 = tio.transforms.Compose([lateral, scaling])
        trans3 = tio.transforms.Compose([affine, scaling])
        trans4 = tio.transforms.Compose([lateral, affine, scaling])
        trans5 = tio.transforms.Compose([lateral, elastic])
        trans6 = tio.transforms.Compose([affine, elastic])
        trans7 = tio.transforms.Compose([scaling, elastic])
        trans8 = tio.transforms.Compose([lateral, affine, elastic])
        trans9 = tio.transforms.Compose([lateral, scaling, elastic])
        trans10 = tio.transforms.Compose([affine, scaling, elastic])
        trans11 = tio.transforms.Compose([lateral, affine, scaling, elastic])

        # create a dict of transformations with probabilities
        transforms_dict = {
            trans1: 1,
            trans2: 1,
            trans3: 1,
            trans4: 1,
            trans5: 1,
            trans6: 1,
            trans7: 1,
            trans8: 1,
            trans9: 1,
            trans10: 1,
            trans11: 1,
            lateral: 1,
            affine: 1,
            scaling: 1,
            elastic: 1,
            none: 15  # don't transform all data
        }
        all_transforms = tio.OneOf(transforms_dict)

        if augmentation:
            new_dataset = tio.SubjectsDataset(self.subjects, transform=all_transforms)
        else:
            new_dataset = tio.SubjectsDataset(self.subjects)

        set_split = 0.2
        dataset_size = len(new_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(set_split * dataset_size))
        train_indices, test_indices = indices[split:], indices[:split]

        train_set = Subset(new_dataset, train_indices)
        test_set = Subset(new_dataset, test_indices)

        return train_set, test_set
