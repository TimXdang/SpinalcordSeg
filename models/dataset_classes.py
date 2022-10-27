from abc import ABC
from torch.utils.data import Dataset


class SpinalCordDataset(Dataset, ABC):
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
