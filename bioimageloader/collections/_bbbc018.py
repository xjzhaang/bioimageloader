from functools import cached_property, partial
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import albumentations
import numpy as np
import tifffile
from PIL import Image

from ..base import NucleiDataset
from ..types import BundledPath
from ..utils import bundle_list, stack_channels, stack_channels_to_rgb


class BBBC018(NucleiDataset):
    """Human HT29 colon-cancer cells (diverse phenotypes)

    The image set consists of 56 fields of view (4 from each of 14 samples).
    Because there are three channels, there are 168 image files. (The samples
    were stained with Hoechst 33342, pH3, and phalloidin. Hoechst 33342 is a DNA
    stain that labels the nucleus. Phospho-histone H3 indicates mitosis.
    Phalloidin labels actin, which is present in the cytoplasm.) The samples are
    the top-scoring sample from each of Jones et al.'s classifiers, as listed in
    the file SamplesScores.zip in their supplement. The files are in DIB format,
    as produced by the Cellomics ArrayScan instrument at the Whitehead–MIT
    Bioimaging Center. We recommend using Bio-Formats to read the DIB files.
    Each image is 512 x 512 pixels.

    The filenames are of the form wellidx-channel.DIB, where wellidx is the
    five-digit well index (from Jones et al.'s supplement) and channel is either
    DNA, actin, or pH3, depending on the channel.

    Notes
    -----
    - BBBC018_v1_images/10779 annotation is missing. len(anno_dict) =
      len(file_list) - 1; ind={26}
        [PosixPath('images/bbbc/018/BBBC018_v1_images/10779-DNA.DIB'),
         PosixPath('images/bbbc/018/BBBC018_v1_images/10779-actin.DIB'),
         PosixPath('images/bbbc/018/BBBC018_v1_images/10779-pH3.DIB')]
    - Every DIB has 3 channels (Order = (DNA,actin,pH3)). The second one is the
      object.
    - DNA -> Nuceli
    - Actin -> Cell
    - Annotation is outline one, but every anno is closed so binary_fill_holes
      works fine
    - For some reason annotation is y inverted

    References
    ----------
    .. [1] https://bbbc.broadinstitute.org/BBBC018
    """

    # Dataset's acronym
    acronym = 'BBBC018'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_calls: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'equal',
        # specific to this dataset
        # image_ch: Sequence[str] = ('DNA', 'actin',),
        anno_ch: Sequence[str] = ('DNA',),
        drop_missing_pairs: bool = True,
        **kwargs
                 # anno_ch: str='DNA',
                 # Always good to have
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
        grayscale_mode : {'cv2', 'equal', Sequence[float]} (default: 'equal')
            How to convert to grayscale. If set to 'cv2', it follows opencv
            implementation. Else if set to 'equal', it sums up values along
            channel axis, then divides it by the number of expected channels.
        # image_ch : {'DNA', 'actin'} (default: ('DNA', 'actin'))
        #     Which channel(s) to load as image. Make sure to give it as a
        #     Sequence when choose a single channel.
        anno_ch : {'DNA', 'actin'} (default: ('DNA',))
            Which channel(s) to load as annotation. Make sure to give it as a
            Sequence when choose a single channel.
        drop_missing_pairs : bool (default: True)
            Valid only if `output='both'`. It will drop images that do not have
            mask pairs.

        See Also
        --------
        NucleiDataset : Super class
        DatasetInterface : Interface
        """
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_calls = num_calls
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # self.image_ch = image_ch
        self.anno_ch = anno_ch
        self.drop_missing_pairs = drop_missing_pairs

        if self.output == 'both' and self.drop_missing_pairs:
            self.file_list, self.anno_dict = self._drop_missing_pairs()

    @classmethod
    def _imread_handler(cls, p: Path) -> np.ndarray:
        img = Image.open(p)
        return np.asarray(img)[..., 1]

    def get_image(self, p: BundledPath) -> np.ndarray:
        # Second channel has objects
        # Order = (DNA,actin,pH3)
        img = stack_channels_to_rgb(self._imread_handler, p, 2, 0, 1)
        return img

    def get_mask(self, p: Union[Path, BundledPath]) -> np.ndarray:
        if isinstance(p, Path):
            mask = np.asarray(Image.open(p))
        else:
            mask = stack_channels(Image.open, p)
        # For some reason mask is -y
        return np.ascontiguousarray(mask[::-1, ...])

    @cached_property
    def file_list(self) -> List[BundledPath]:
        root_dir = self.root_dir
        parent = 'BBBC018_v1_images'
        # Order = (DNA,actin,pH3)
        _file_list = sorted(root_dir.glob(f'{parent}/*.DIB'))
        return bundle_list(_file_list, 3)

    @cached_property
    def anno_dict(self) -> Dict[int, Union[Path, BundledPath]]:
        anno_dict: Dict[int, Union[Path, BundledPath]] = {}
        root_dir = self.root_dir
        parent = 'BBBC018_v1_outlines'
        # _anno_list = sorted(root_dir.glob(f'{parent}/*.png'))
        stain_to_target = {'DNA': 'nuclei',
                           'actin': 'cells'}
        if len(ch := self.anno_ch) == 1:
            target = stain_to_target[ch[0]]
            for i, p in enumerate(self.file_list):
                name = p[0].stem.split('-')[0]
                fn = root_dir / parent / f'{name}-{target}.png'
                if fn.exists():
                    anno_dict[i] = fn
        elif len(ch) == 2:
            for i, p in enumerate(self.file_list):
                name = p[0].stem.split('-')[0]
                lst_fn = []
                for c in ch:
                    target = stain_to_target[c]
                    fn = root_dir / parent / f'{name}-{target}.png'
                    if fn.exists():
                        lst_fn.append(fn)
                if lst_fn:
                    anno_dict[i] = lst_fn
        else:
            raise ValueError
        return anno_dict
