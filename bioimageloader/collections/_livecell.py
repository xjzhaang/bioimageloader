from functools import cached_property
from pathlib import Path
import pathlib
from typing import Dict, List, Optional, Sequence, Union, Any

import albumentations
import cv2
import numpy as np
import tifffile
from pycocotools import coco

from ..base import MaskDataset
from ..types import BundledPath
from ..utils import imread_asarray, rle_decoding_inseg, read_csv, ordered_unique

class LIVECell(MaskDataset):
    """LIVECEll
    A large-scale dataset for label-free live cell segmentation

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

    References
    ----------
    .. [1] https://sartorius-research.github.io/LIVECell/
    .. [2] https://www.nature.com/articles/s41592-021-01249-6

    See Also
    --------
    MaskDataset : Super class
    Dataset : Base class
    DatasetInterface : Interface

    """
    # Set acronym
    acronym = 'LIVECell'

    def __init__(
        self,
        root_dir: str,
        *,
        output: str = 'both',
        transforms: Optional[albumentations.Compose] = None,
        num_samples: Optional[int] = None,
        grayscale: bool = False,
        grayscale_mode: Union[str, Sequence[float]] = 'cv2',
        # specific to this dataset
        training: bool = True,
        **kwargs
    ):
        self._root_dir = root_dir
        self._output = output
        self._transforms = transforms
        self._num_samples = num_samples
        self._grayscale = grayscale
        self._grayscale_mode = grayscale_mode
        # specific to this one here
        self.training = training
        
        #Read training/val or test annotations from json
        if self.training:
            self.coco_tr = coco.COCO(root_dir + "/livecell_coco_train.json")
            self.coco_val = coco.COCO(root_dir + "/livecell_coco_val.json")
            img_tr = self.coco_tr.loadImgs(self.coco_tr.getImgIds())
            img_va = self.coco_val.loadImgs(self.coco_val.getImgIds())
            self.anno_dictionary = img_tr + img_va
        else:
            self.coco_te = coco.COCO(root_dir + "/livecell_coco_test.json")
            img_te = self.coco_te.loadImgs(self.coco_te.getImgIds())
            self.anno_dictionary = img_te 
            
    
    def get_image(self, p: Path) -> np.ndarray:
        img = tifffile.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def get_mask(self, p: Path) -> np.ndarray:
        #find the annotation id associated with the file name
        #then get the mask
        if self.training:
            anno = list(filter(lambda img: img['file_name'] == str(pathlib.PurePath(p)).split('/')[-1], self.anno_dictionary))
            try:
                annIds = self.coco_tr.getAnnIds(imgIds=anno[0]["id"], iscrowd=None)
                anns = self.coco_tr.loadAnns(annIds)
                mask = self.coco_tr.annToMask(anns[0])
                for i in range(len(anns)):
                    mask += self.coco_tr.annToMask(anns[i])
            except:
                annIds = self.coco_val.getAnnIds(imgIds=anno[0]["id"], iscrowd=None)
                anns = self.coco_val.loadAnns(annIds)
                mask = self.coco_val.annToMask(anns[0])
                for i in range(len(anns)):
                    mask += self.coco_val.annToMask(anns[i])
        else:
            anno = list(filter(lambda img: img['file_name'] == str(pathlib.PurePath(p)).split('/')[-1], self.anno_dictionary))       
            annIds = self.coco_te.getAnnIds(imgIds=anno[0]["id"], iscrowd=None)
            anns = self.coco_te.loadAnns(annIds)
            mask = self.coco_te.annToMask(anns[0])
            for i in range(len(anns)):
                mask += self.coco_te.annToMask(anns[i])
        return mask

    @cached_property
    def file_list(self) -> List[Path]:
        # Call MaskDataset.root_dir
        root_dir = self.root_dir
        parent = 'images/livecell_train_val_images' if self.training else 'images/livecell_test_images'
        return sorted(root_dir.glob(f'{parent}/*.tif'))

    @cached_property
    def anno_dict(self) -> List[Path]:
        root_dir = self.root_dir
        parent = 'images/livecell_train_val_images' if self.training else 'images/livecell_test_images'
        return sorted(root_dir.glob(f'{parent}/*.tif'))
       
            
            
