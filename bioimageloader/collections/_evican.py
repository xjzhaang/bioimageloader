from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np
from PIL import Image
import os, glob


from ..base import MaskDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels, stack_channels_to_rgb

class EVICAN(MaskDataset):
    """EVICAN
    A balanced dataset for algorithm development in cell and nucleus segmentation 
    
    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'both', 'image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines augmentation in
        sequence.
    num_samples : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    grayscale : bool, default: False
        Convert images to grayscale
    grayscale_mode : {'cv2', 'equal', Sequence[float]}, default: 'cv2'
        How to convert to grayscale. If set to 'cv2', it follows opencv
        implementation. Else if set to 'equal', it sums up values along channel
        axis, then divides it by the number of expected channels.
    training : bool, default: True
        Load training set if True, else load testing one
    anno_ch : {'cell', 'nuclei'}, default: 'cell'
        Specify which annotation mask to use
    
    References
    ----------
    .. [1] https://academic.oup.com/bioinformatics/article/36/12/3863/5814923
    
    
    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface
    """
    # Set acronym
    acronym = 'EVICAN'

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,  # optional
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',  # optional
        # specific to this dataset
        training: bool = True,
        anno_ch: str = 'cells',
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale   # optional
        self._grayscale_mode = grayscale_mode  # optional
        # specific to this one here
        self.training = training
        self.anno_ch = anno_ch
        if anno_ch != 'nuclei' and anno_ch != 'cells':
            raise ValueError("Set `anno_ch` either 'nuclei' or 'cells'")
            
        for p in Path(self.root_dir).glob("Images/EVICAN_val2019/*Background.jpg"):
            p.unlink()
        for p in Path(self.root_dir).glob("Images/EVICAN_train2019/*Background.jpg"):
            p.unlink()

    def get_image(self, p: Path) -> np.ndarray:
        img = Image.open(p)
        return np.asarray(img)

    def get_mask(self, p: Path) -> np.ndarray:
        mask = Image.open(p)
        # dtype=bool originally and bool is not well handled by albumentations
        mask = mask.convert(mode='1')
        return 255 * np.asarray(mask)

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        if self.training:
            parent = 'Images/EVICAN_train2019' 
            file_list = sorted(root_dir.glob(f'{parent}/*.jpg'))
        else:
            parent = 'Images/EVICAN_val2019'
            file_list = sorted(root_dir.glob(f'{parent}/*.jpg'))
        return file_list

    @cached_property
    def anno_dict(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        if self.anno_ch == "cells":
            if self.training:
                parent = 'Masks/EVICAN_train_masks' 
                file_list = sorted(root_dir.glob(f'{parent}/Cells/*.jpg'))
            else:
                parent = 'Masks/EVICAN_val_masks'
                file_list = sorted(root_dir.glob(f'{parent}/Cells/*.jpg'))
        if self.anno_ch == "nuclei":
            if self.training:
                parent = 'Masks/EVICAN_train_masks' 
                file_list = sorted(root_dir.glob(f'{parent}/Nuclei/*.jpg'))
            else:
                parent = 'Masks/EVICAN_val_masks'
                file_list = sorted(root_dir.glob(f'{parent}/Nuclei/*.jpg'))
        return file_list
