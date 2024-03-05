import os
import numpy as np

from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .dataset import CustomImageFolder

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CIFAR10_CLS_TO_IDX = {cls:idx for idx, cls in enumerate(CIFAR10_CLASSES)}

def auto_find_path(root, data_dir, file_name):
    data_path = os.path.join(root, file_name)
    if os.path.isfile(data_path):
        return root, data_path

    data_dir = os.path.join(root, data_dir)
    data_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(data_path):
        raise Exception(f'!failed to find {file_name} under {root} or {data_dir}.')

    return data_dir, data_path
        
class CIFAR10_1(Dataset):

    NAME = 'CIFAR10.1'
    DEFAULT_ROOT_DIR = 'cifar-10.1'
    DATA_FILES = {
        'v4' : ('cifar10.1_v4_data.npy', 'cifar10.1_v4_labels.npy'),
        'v6' : ('cifar10.1_v6_data.npy', 'cifar10.1_v6_labels.npy')
    }
    
    def __init__(self, root, version='v6', transform=None, target_transform=None):
        assert version in self.DATA_FILES

        data_fname, target_fname = self.DATA_FILES[version]
        root, data_path = auto_find_path(root, self.DEFAULT_ROOT_DIR, data_fname)        
        target_path = os.path.join(root, target_fname)

        data = np.load(data_path)
        target = np.load(target_path)

        assert data.shape[0] == target.shape[0]
        true_size = 2021 if version == 'v4' else 2000
        assert data.shape[0] == true_size

        self.data, self.target = data, target
        self.transform, self.target_transform = transform, target_transform
        
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data, target = self.data[idx], self.target[idx]
        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target.item()


class CIFAR10_2(Dataset):
    NAME = 'CIFAR10.2'
    DEFAULT_ROOT_DIR = 'cifar-10.2'
    DATA_FILES = {
        'train' : 'cifar102_train.npz',
        'test' : 'cifar102_test.npz'
    }
    
    def __init__(self, root, split='test', transform=None, target_transform=None):
        _, data_path = auto_find_path(root, self.DEFAULT_ROOT_DIR, self.DATA_FILES[split])
        data = np.load(data_path)
        self.images = data['images']
        self.targets = data['labels']

        true_size = 10000 if split == 'train' else 2000
        assert self.images.shape[0] == true_size

        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image, target = self.images[idx], self.targets[idx]
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target.item()
        
class CINIC10_(ImageFolder):
    SPLITS = ['train', 'val', 'test']

    DATA_DIR = 'CINIC-10'

    MEAN = [0.47889522, 0.47227842, 0.43047404]
    STD = [0.24205776, 0.23828046, 0.25874835]
    
    def __init__(self, root, split='test', transform=None, target_transform=None):
        assert split in self.SPLITS

        data_dir = os.path.join(root, split)
        if not os.path.isdir(data_dir):
            root = os.path.join(root, self.DATA_DIR)
            data_dir = os.path.join(root, split)

            assert os.path.isdir(data_dir)

        filter_cifar10_test_data = lambda x: 'cifar10' not in x
        
        super().__init__(data_dir,
                         transform=transform,
                         target_transform=target_transform,
                         is_valid_file=filter_cifar10_test_data)

class CINIC10(CustomImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(os.path.join(root, 'CINIC-10/test'),
                         transform=transform,
                         target_transform=target_transform,
                         class_to_idx=CIFAR10_CLS_TO_IDX,
                         data_list='image_ids/CINIC-10_image_ids.txt')

        
class CIFAR10_R(CustomImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(os.path.join(root, 'cifar-10-r'),
                         transform=transform,
                         target_transform=target_transform,
                         class_to_idx=CIFAR10_CLS_TO_IDX,
                         data_list='image_ids/CIFAR10-R_image_ids.txt')
        
DATASETS = {
    'cifar10.1' : CIFAR10_1,
    'cifar10.2' : CIFAR10_2,
    'cinic' : CINIC10,
    'cifar10-r': CIFAR10_R
}
