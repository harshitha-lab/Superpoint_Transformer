import os
import sys
import glob
import torch
import shutil
import logging
import pandas as pd
import os.path as osp
from src.datasets import BaseDataset
from src.data import Data, Batch, InstanceData
from src.datasets.segment3d_config import *
from torch_geometric.data import extract_zip
from src.utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb
from src.transforms import RoomPosition


DIR = osp.dirname(osp.realpath(__file__))
log = logging.getLogger(__name__)


__all__ = ['SEGMENT3D', 'MiniSEGMENT3D']


########################################################################
#                                 Utils                                #
########################################################################


def read_las_tile(
        filepath, 
        xyz=True, 
        rgb=True, 
        intensity=False, 
        semantic=True, 
        instance=False,
        remap=False, 
        max_intensity=600):
    
    import laspy
    
    """Read a tile saved as LAS.

    :param filepath: str
        Absolute path to the LAS file
    :param xyz: bool
        Whether XYZ coordinates should be saved in the output Data.pos
    :param rgb: bool
        Whether RGB colors should be saved in the output Data.rgb
    :param intensity: bool
        Whether intensity should be saved in the output Data.rgb
    :param semantic: bool
        Whether semantic labels should be saved in the output Data.y
    :param instance: bool
        Whether instance labels should be saved in the output Data.obj
    :param remap: bool
        Whether semantic labels should be mapped from their Vancouver ID
        to their train ID
    :param max_intensity: float
        Maximum value used to clip intensity signal before normalizing 
        to [0, 1]
    """
    # Create an emty Data object
    data = Data()
    
    las = laspy.read(filepath)

    # Populate data with point coordinates 
    if xyz:
        # Apply the scale provided by the LAS header
        pos = torch.stack([
            torch.tensor(las[axis].copy()   )
            for axis in ["X", "Y", "Z"]], dim=-1)
        pos *= las.header.scale
        pos_offset = pos[0]
        data.pos = (pos - pos_offset).float()
        data.pos_offset = pos_offset

    # Populate data with point RGB colors
    if rgb:
        # RGB stored in uint16 lives in [0, 65535]
        data.rgb = to_float_rgb(torch.stack([
            torch.FloatTensor(las[axis].astype('float32') / 65535)
            for axis in ["red", "green", "blue"]], dim=-1))

    # # Populate data with point LiDAR intensity
    # if intensity:
    #     # Heuristic to bring the intensity distribution in [0, 1]
    #     data.intensity = torch.FloatTensor(
    #         las['intensity'].astype('float32')
    #     ).clip(min=0, max=max_intensity) / max_intensity

    # Populate data with point semantic segmentation labels
    if semantic:
        y = torch.LongTensor(las['classification'])
        data.y = torch.from_numpy(THING_CLASSES)[y] if remap else y

    # Populate data with point panoptic segmentation labels
    if instance:
        raise NotImplementedError("The dataset does not contain instance labels.")

    print("data::: ",data , flush=True)
    return data

########################################################################
#                               SEGMENT3D                               #
########################################################################

class SEGMENT3D(BaseDataset):
    """SEGMENT3D dataset, for Area-wise prediction.

    Note: we are using the SEGMENT3D version with non-aligned rooms, which
    contains `Area_{{i_area:1>6}}_alignmentAngle.txt` files. Make sure
    you are not using the aligned version.

    Dataset website: http://buildingparser.stanford.edu/dataset.html

    Parameters
    ----------
    root : `str`
        Root directory where the dataset should be saved.
    fold : `int`
        Integer in [1, ..., 6] indicating the Test Area
    with_stuff: `bool`
        By default, SEGMENT3D does not have any stuff class. If `with_stuff`
        is True, the 'ceiling', 'wall', and 'floor' classes will be
        treated as stuff
    stage : {'train', 'val', 'test', 'trainval'}, optional
    transform : `callable`, optional
        transform function operating on data.
    pre_transform : `callable`, optional
        pre_transform function operating on data.
    pre_filter : `callable`, optional
        pre_filter function operating on data.
    on_device_transform: `callable`, optional
        on_device_transform function operating on data, in the
        'on_after_batch_transfer' hook. This is where GPU-based
        augmentations should be, as well as any Transform you do not
        want to run in CPU-based DataLoaders
    """

    _form_url = FORM_URL
    _zip_name = ZIP_NAME
    _aligned_zip_name = ALIGNED_ZIP_NAME
    _unzip_name = UNZIP_NAME

    def __init__(self, *args, fold=5, with_stuff=False, **kwargs):
        self.fold = fold
        self.with_stuff = with_stuff
        super().__init__(*args, val_mixed_in_train=False, **kwargs)

    @property
    def pre_transform_hash(self):
        """Produce a unique but stable hash based on the dataset's
        `pre_transform` attributes (as exposed by `_repr`).

        For SEGMENT3D, we want the hash to detect if the stuff classes are
        the default ones.
        """
        suffix = '_with_stuff' if self.with_stuff else ''
        return super().pre_transform_hash + suffix

    @property
    def class_names(self):
        """List of string names for dataset classes. This list must be
        one-item larger than `self.num_classes`, with the last label
        corresponding to 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return CLASS_NAMES

    @property
    def num_classes(self):
        """Number of classes in the dataset. Must be one-item smaller
        than `self.class_names`, to account for the last class name
        being used for 'void', 'unlabelled', 'ignored' classes,
        indicated as `y=self.num_classes` in the dataset labels.
        """
        return SEGMENT3D_NUM_CLASSES

    @property
    def stuff_classes(self):
        """List of 'stuff' labels for INSTANCE and PANOPTIC
        SEGMENTATION (setting this is NOT REQUIRED FOR SEMANTIC
        SEGMENTATION alone). By definition, 'stuff' labels are labels in
        `[0, self.num_classes-1]` which are not 'thing' labels.

        In instance segmentation, 'stuff' classes are not taken into
        account in performance metrics computation.

        In panoptic segmentation, 'stuff' classes are taken into account
        in performance metrics computation. Besides, each cloud/scene
        can only have at most one instance of each 'stuff' class.

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc), while
        `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        """
        return STUFF_CLASSES_MODIFIED if self.with_stuff else STUFF_CLASSES

    @property
    def class_colors(self):
        """Colors for visualization, if not None, must have the same
        length as `self.num_classes`. If None, the visualizer will use
        the label values in the data to generate random colors.
        """
        return CLASS_COLORS

    @property
    def all_base_cloud_ids(self):
        """Dictionary holding lists of clouds ids, for each
        stage.

        The following structure is expected:
            `{'train': [...], 'val': [...], 'test': [...]}`
        """
        return    {
            'train': ['esterhazy_k1_NewP2_2_33_classified2_1.las','esterhazy_k1_NewP2_2_33_classified2_3_21.las'],
            'val': ['esterhazy_k1_NewP2_2_33_classified2_2.las'],
            'test':  ['esterhazy_k1_NewP2_2_33_classified2_3_1.las']
            }
        # return    {
        #     'train': ['DrumPlant_Tank_rotate45_classified8.las'],
        #     'val': ['DrumPlant_Tank_rotate45_classified_sample1.las', 'DrumPlant_Tank_rotate45_classified_sample2.las'],
        #     'test':  ['DrumPlant_Tank_rotate45_classified4.las'] 
        #     }
    
    # {
    #         'train': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
    #         'val': [f'Area_{i}' for i in range(1, 7) if i != self.fold],
    #         'test': [f'Area_{self.fold}']}

    def download_dataset(self):
        """Download the SEGMENT3D dataset.
        """
        # Manually download the dataset
        if not osp.exists(osp.join(self.root, self._zip_name)):
            log.error(
                f"\nSEGMENT3D does not support automatic download.\n"
                f"Please, register yourself by filling up the form at "
                f"{self._form_url}\n"
                f"From there, manually download the non-aligned rooms"
                f"'{self._zip_name}' into your '{self.root}/' directory and "
                f"re-run.\n"
                f"The dataset will automatically be unzipped into the "
                f"following structure:\n"
                f"{self.raw_file_structure}\n"
                f"⛔ Make sure you DO NOT download the "
                f"'{self._aligned_zip_name}' version, which does not contain "
                f"the required `Area_{{i_area:1>6}}_alignmentAngle.txt` files."
                f"\n")
            sys.exit(1)

        # Unzip the file and rename it into the `root/raw/` directory. This
        # directory contains the raw Area folders from the zip
        extract_zip(osp.join(self.root, self._zip_name), self.root)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self._unzip_name), self.raw_dir)

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw cloud and return a `Data` object, ready to
        be passed to `self.pre_transform`.

        This `Data` object should contain the following attributes:
          - `pos`: point coordinates
          - `y`: OPTIONAL point semantic label
          - `obj`: OPTIONAL `InstanceData` object with instance labels
          - `rgb`: OPTIONAL point color
          - `intensity`: OPTIONAL point LiDAR intensity

        IMPORTANT:
        By convention, we assume `y ∈ [0, self.num_classes-1]` ARE ALL
        VALID LABELS (i.e. not 'ignored', 'void', 'unknown', etc),
        while `y < 0` AND `y >= self.num_classes` ARE VOID LABELS.
        This applies to both `Data.y` and `Data.obj.y`.
        """
        print("raw_cloud_path::: ",raw_cloud_path , flush=True)
        print("os.getcwd::: ",os.getcwd() , flush=True)
        # import glob
        
        # folder_path = '../../../InputData/INPUT_pointcloudData/raw/'


        # filepaths = glob.glob(os.path.join(folder_path, '*.las'))
        # filepath = filepaths[0] if isinstance(filepaths, list) and filepaths else filepaths 
        return read_las_tile(raw_cloud_path[:-4])
    
    
    # read_SEGMENT3D_area(
    #         raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=True,
    #         xyz_room=True, align=False, is_val=True, verbose=False)

    # @property
    # def raw_file_structure(self):
    #     return f"""
    # {self.root}/
    #     └── {self._zip_name}
    #     └── raw/
    #         └── Area_{{i_area:1>6}}/
    #             └── Area_{{i_area:1>6}}_alignmentAngle.txt
    #             └── ...
    #         """

    @property
    def raw_file_names(self):
        """The file paths to find in order to skip the download."""
        area_folders = super().raw_file_names
        alignment_files = [
            osp.join(a, f"{a}_alignmentAngle.txt") for a in area_folders]
        return area_folders + alignment_files

    def id_to_relative_raw_path(self, id):
        """Given a cloud id as stored in `self.cloud_ids`, return the
        path (relative to `self.raw_dir`) of the corresponding raw
        cloud.
        """
        return self.id_to_base_id(id)


########################################################################
#                              MiniSEGMENT3D                               #
########################################################################

class MiniSEGMENT3D(SEGMENT3D):
    """A mini version of SEGMENT3D with only 1 area per stage for
    experimentation.
    """
    _NUM_MINI = 1

    @property
    def all_cloud_ids(self):
        return {k: v[:self._NUM_MINI] for k, v in super().all_cloud_ids.items()}

    @property
    def data_subdir_name(self):
        return self.__class__.__bases__[0].__name__.lower()

    # We have to include this method, otherwise the parent class skips
    # processing
    def process(self):
        super().process()

    # We have to include this method, otherwise the parent class skips
    # processing
    def download(self):
        super().download()
