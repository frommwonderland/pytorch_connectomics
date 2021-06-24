from typing import Optional, List
import numpy as np

import torch
import torch.utils.data
from dataset_volume import VolumeDataset
from ..augmentation import Compose
from ..utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


class PairDataset(VolumeDataset):
    r""" This Dataloader will prepare sample that are pairs for feeding the contrastive
    learning algorithm.

    Args:
        volume (list): list of image volumes.
        label (list, optional): list of label volumes. Default: None
        valid_mask (list, optional): list of valid masks. Default: None
        valid_ratio (float): volume ratio threshold for valid samples. Default: 0.5
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: None
    """
    def __init__(self,
                 volume: list,
                 label: Optional[list] = None,
                 valid_mask: Optional[list] = None,
                 valid_ratio: float = 0.5,
                 sample_volume_size: tuple = (129, 129, 129),
                 sample_label_size: tuple = (129, 129, 129),
                 sample_stride: tuple = (1, 1, 1),
                 augmentor: AUGMENTOR_TYPE = None,
                 mode: str = 'train',
                 iter_num: int = -1,
                 data_mean=0.5,
                 data_std=0.5):

        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.volume = volume
        self.label = label
        self.augmentor = augmentor
        self.sample_volume_size = np.array(sample_volume_size).astype(int)

        print('---------------            I am Pairwise            ----------------')

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample_pair = self._create_sample_pair()

    def _create_sample_pair(self):
        r"""Create a sample pair that will be used for contrastive learning.
        """
        sample = self._random_sampling(self.sample_volume_size)
        pos, out_volume, out_label, out_valid = sample
