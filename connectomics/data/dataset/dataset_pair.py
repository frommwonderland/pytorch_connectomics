from typing import Optional, List
import numpy as np

import torch
import torch.utils.data
from .dataset_volume import VolumeDataset
from ..augmentation import Compose
from ..utils import *

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]

class PairDataset(VolumeDataset):
    r""" This Dataloader will prepare sample that are pairs for feeding the contrastive
    learning algorithm.
    Inherits all the attributes and functions from parent VolumeDataset.
    """
    def __init__(self,
                 volume: list,
                 label: Optional[list] = None,
                 valid_mask: Optional[list] = None,
                 valid_ratio: float = 0.5,
                 sample_volume_size: tuple = (8, 64, 64),
                 sample_label_size: tuple = (8, 64, 64),
                 sample_stride: tuple = (1, 1, 1),
                 augmentor: AUGMENTOR_TYPE = None,
                 target_opt: TARGET_OPT_TYPE = ['1'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 erosion_rates: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 mode: str = 'train',
                 do_2d: bool = False,
                 iter_num: int = -1,
                 reject_size_thres: int = 0,
                 reject_diversity: int = 0,
                 reject_p: float = 0.95,
                 data_mean=0.5,
                 data_std=0.5):

        super().__init__(volume, label, valid_mask, valid_ratio, sample_volume_size, sample_label_size, sample_stride,
        augmentor, target_opt, weight_opt, erosion_rates, dilation_rates, mode, do_2d, iter_num)

        self.num_augmented_images = 2

    def __len__(self):
        pass

    def __getitem__(self, idx):
        if self.mode == 'train':
            sample_pair = self._create_sample_pair()
            return sample_pair

    def _create_sample_pair(self):
        r"""Create a sample pair that will be used for contrastive learning.
        """
        sample_pair = list()

        sample = self._random_sampling(self.sample_volume_size)
        pos, out_volume, out_label, out_valid = sample
        out_volume = self._create_masked_input(out_volume, out_label)

        data = {'image': out_volume}
        for i in range(self.num_augmented_images):
            augmented = self.augmentor(data)
            sample_pair.append(augmented['image'])

        return sample_pair

    def _create_masked_input(self, vol: np.ndarray, label: np.ndarray) -> np.ndarray:
        r"""Create masked input volume, that is pure EM where the mask is not 0. Otherwise all
        values set to 0. Returns the prepared mask.
        Args:
            vol (numpy.ndarray): volume that is EM input.
            label (numpy.ndarray): associated label volume.
        """
        vol[np.where(label == 0)] = 0
        return vol
