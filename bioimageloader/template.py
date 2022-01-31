from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np

from bioimageloader.base import NucleiDataset


class Template(NucleiDataset):
    """Template
    """

    # Set acronym
    acronym = ''

    def __init__(
        self,
        root_dir: str,
        *,  # only keyword param
        output: str = 'both',  # optional
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,  # optional
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',  # optional
        # specific to this dataset
        **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory
        output : {'image', 'mask', 'both'} (default: 'both')
            Change outputs. 'both' returns {'image': image, 'mask': mask}.
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.
        grayscale : bool (default: False)
            Convert images to grayscale
        grayscale_mode : {'cv2', 'equal', Sequence[float]} (default: 'cv2')
            How to convert to grayscale. If set to 'cv2', it follows opencv
            implementation. Else if set to 'equal', it sums up values along
            channel axis, then divides it by the number of expected channels.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface

        """
        self._root_dir = root_dir
        self._output = output  # optional
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale   # optional
        self._grayscale_mode = grayscale_mode  # optional
        # specific to this one here

    def get_image(self, p: Path) -> np.ndarray:
        ...

    # optional
    def get_mask(self, p: Path) -> np.ndarray:
        ...

    @cached_property
    def file_list(self) -> Union[List[Path], List[List[Path]]]:
        # Important to decorate with `cached_property` in general
        ...

    # optional
    @cached_property
    def anno_dict(self) -> Dict[int, Union[Path, List[Path]]]:
        # Important to decorate with `cached_property` in general
        ...
