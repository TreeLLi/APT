import os

from .dataset import CustomImageFolder

from addict import Dict

DATASETS = Dict()

DATASETS[None].data_dir = 'imagenet'
DATASETS[None].data_list = 'image_ids/imagenet_test_image_ids.txt'

DATASETS.R.data_dir = 'imagenet-r'
DATASETS.R.data_list = 'image_ids/imagenet-r_image_ids.txt'

DATASETS.A.data_dir = 'imagenet-a'
DATASETS.A.data_list = 'image_ids/imagenet-a_image_ids.txt'

DATASETS.v2.data_dir = 'imagenetv2-matched-frequency-format-val'
DATASETS.v2.data_list = 'image_ids/imagenet-v2_image_ids.txt'

DATASETS.ON.data_dir = 'objectnet'
DATASETS.ON.data_list = 'image_ids/objectnet_image_ids.txt'


class ImageNet(CustomImageFolder):
    VARIANTS = list(DATASETS.keys())
    VARIANTS.remove(None)

    def __init__(self, root, variant=None, split=None, transform=None, target_transform=None):
        data_dir = os.path.join(root, DATASETS[variant].data_dir)
        split = split if variant == None else None # no split for non-base variants
        if split is not None: data_dir = os.path.join(data_dir, split)
        assert os.path.isdir(data_dir)

        if variant in ['R', 'A', 'ON']:
            # remap targets to ImageNet class idx
            class_to_idx = ImageNet(root, split='val').class_to_idx
        elif variant == 'v2':
            class_to_idx = {str(i):i for i in range(1000)}
        else:
            class_to_idx = None

        super().__init__(data_dir,
                         transform=transform,
                         target_transform=target_transform,
                         class_to_idx=class_to_idx,
                         data_list=DATASETS[variant].data_list)
