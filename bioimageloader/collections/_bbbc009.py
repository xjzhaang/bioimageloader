from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import cv2
import numpy as np
import tifffile


from ..base import MaskDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels, stack_channels_to_rgb

class BBBC009(MaskDataset):
    """Human red blood cells
    
    This image set consists of five differential interference contrast (DIC) images of red bood cells. 
    
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
        
    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC009
    
    
    See Also
    --------
    MaskDataset : Super class
    DatasetInterface : Interface
    """
    # Set acronym
    acronym = 'BBBC009'

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
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale   # optional
        self._grayscale_mode = grayscale_mode  # optional
        # specific to this one here

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = tifffile.imread(p)
        # dtype=bool originally and bool is not well handled by albumentations
        return mask.astype(np.uint8)

    @cached_property
    def file_list(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'human_rbc_dic_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> List[Path]:
        # Important to decorate with `cached_property` in general
        root_dir = self.root_dir
        parent = 'human_rbc_dic_outlines'
        file_list = sorted(root_dir.glob(f'{parent}/*.tif'))
        return file_list