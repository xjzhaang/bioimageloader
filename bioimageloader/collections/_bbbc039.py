from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import albumentations
import numpy as np
import tifffile
from PIL import Image

from ..base import MaskDataset


class BBBC039(MaskDataset):
    """Nuclei of U2OS cells in a chemical screen [1]_

    This data set has a total of 200 fields of view of nuclei captured with
    fluorescence microscopy using the Hoechst stain. These images are a sample
    of the larger BBBC022 chemical screen. The images are stored as TIFF files
    with 520x696 pixels at 16 bits.

    Parameters
    ----------
    root_dir : str
        Path to root directory
    output : {'both', 'image', 'mask'}, default: 'both'
        Change outputs. 'both' returns {'image': image, 'mask': mask}.
    transforms : albumentations.Compose, optional
        An instance of Compose (albumentations pkg) that defines augmentation in
        sequence.
    num_calls : int, optional
        Useful when ``transforms`` is set. Define the total length of the
        dataset. If it is set, it overwrites ``__len__``.
    training : bool or list of int
        Load training data if True, else load testing data.

    Notes
    -----
    - Split (training/valiadation/test)
        - `training=True` combines 'training' with 'validation'
    - Sample of larger BBBC022 and did manual segmentation
    - Overlap some with DSB2018
    - Mask is png but (instance) value is only stored in RED channel
    - Maximum value is 2**12

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC039

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """

    # Dataset's acronym
    acronym = 'BBBC039'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        # specific to this dataset
        training: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        # specific to this dataset
        self.training = training

    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = (img / 2**4).astype(np.uint8)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        mask = np.asarray(Image.open(p))[..., 0]
        return mask > 0

    @cached_property
    def file_list(self) -> List[Path]:
        root_dir = self.root_dir
        parent = root_dir / 'images'
        file_list = []
        for name in self.ids:
            p = parent / name
            file_list.append(p.with_suffix('.tif'))
        return file_list

    @cached_property
    def anno_dict(self) -> Dict[int, Path]:
        root_dir = self.root_dir
        parent = root_dir / 'masks'
        anno_list = []
        for name in self.ids:
            p = parent / name
            anno_list.append(p)
        return dict((k, v) for k, v in enumerate(anno_list))

    @cached_property
    def ids(self) -> list:
        def _readlines(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            return list(map(lambda s: s.strip(), lines))
        meta_dir = self.root_dir / 'metadata'
        if self.training:
            # Combine training and validation
            meta_file = meta_dir / 'training.txt'
            _ids = _readlines(meta_file)
            meta_file = meta_dir / 'validation.txt'
            _ids += _readlines(meta_file)
        else:
            meta_file = meta_dir / 'test.txt'
            _ids = _readlines(meta_file)
        return _ids
