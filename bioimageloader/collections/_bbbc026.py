from functools import cached_property
from pathlib import Path
from typing import List, Optional

import albumentations
import cv2
import numpy as np

from ..base import NucleiDataset
from ..utils import imread_asarray


class BBBC026(NucleiDataset):
    """Human Hepatocyte and Murine Fibroblast cells – Co-culture experiment

    This 384-well plate has images of co-cultured hepatocytes and fibroblasts.
    Every other well is populated (A01, A03, ..., C01, C03, ...) such that 96
    wells comprise the data. Each well has 9 sites and thus 9 images associated,
    totaling 864 images.

    For each well there is one field and a single image nuclear image (Hoecsht).
    Images are in 8-bit PNG format.

    Notes
    -----
    - Only centers are annotated for 5 imgages (not implemented)

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC026
    """
    # Dataset's acronym
    acronym = 'BBBC026'

    def __init__(
        self,
        root_dir: str,
        *,
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        root_dir : str
            Path to root directory
        transforms : albumentations.Compose, optional
            An instance of Compose (albumentations pkg) that defines
            augmentation in sequence.
        num_calls : int, optional
            Useful when `transforms` is set. Define the total length of the
            dataset. If it is set, it overrides __len__.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface
        """
        self._root_dir = root_dir
        self._transforms = transforms
        self._num_calls = num_calls

    def get_image(self, p: Path) -> np.ndarray:
        img = imread_asarray(p)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'BBBC026_v1_images'
        file_list = sorted(root_dir.glob(f'{parent}/*.png'))
        return file_list
