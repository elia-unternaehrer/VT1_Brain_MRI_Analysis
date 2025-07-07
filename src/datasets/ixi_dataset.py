#%%
import numpy as np
from scipy.ndimage import affine_transform
import nibabel as nib
from torch.utils.data import Dataset
import os, re, torch
import pandas as pd

#%%
def interpolate_to_grid(img_path, mask_path=None, transform_affine=np.eye(4), voxel_space=(1,1,1), target_shape=(256, 256, 256)):
    """
    Resample an MRI scan to fixed shape and 1mm³ voxels using a single affine transform.

    Args:
        img_path (str): Path to the MRI image.
        transform_affine (np.ndarray): affine transformation matrix (rotation/zoom/translation) in world coordinates (default: np.eye(4), for no transformations).
        voxel_space (tuple): Desired voxel size in mm (default: (1, 1, 1)).
        target_shape (tuple): Desired voxel grid size (default: (256, 256, 256)).
    returns:
        np.ndarray: Resampled MRI image.
    """

    # Load image and get data/affine
    img = nib.load(img_path)
    data = img.get_fdata()

    if mask_path is not None:
        mask = nib.load(mask_path)
        mask_data = mask.get_fdata()
        data = data * mask_data

    original_affine = img.affine # affine matrix, that maps voxel coordinates to world coordinates

    #normalize the data
    non_zero = data[data > 0]
    mean = np.mean(non_zero)
    std = np.std(non_zero)
    data = (data - mean) / std

    # ------------------------------------------------------------------------------------------------------
    # we compute the target affine matrix, that maps the target voxel coordinates to world coordinates (values are in world coordinates)
    # the target affine matrix is a diagonal matrix with the desired voxel size in mm in the diagonal
    # and the translation vector is added to the last column to align the target grid with the original image
    # ------------------------------------------------------------------------------------------------------

    # Compute target affine with defined voxel size
    target_affine = np.diag(voxel_space + (1,))
    
    # Compute center of target grid in voxel coordinates
    target_center_voxel = np.array(target_shape) / 2
    
    # transform the target grid center from voxel coordinates to world coordinates
    target_center_world = target_affine @ np.append(target_center_voxel, 1) # append 1, so we are in homogenus coordinates

    # Compute center of original image in voxel coordinates (of the original image)
    source_center_voxel = np.array(data.shape) / 2

    # transform the original image center from its voxel coordinates to world coordinates
    source_center_world = original_affine @ np.append(source_center_voxel, 1)

    # compute the translation vector to move the target grid center to the original image center (the difference between the two centers)
    translation_vec = source_center_world[:3] - target_center_world[:3]

    # add the translation to the target affine matrix to align the target grid with the original image
    # so that when we apply the target affine matrix to the center of the target grid, we get the center of the original image (in world coordinates)
    target_affine[:3, 3] = translation_vec

    # ------------------------------------------------------------------------------------------------------
    # Building the full affine matrix that maps the target voxel coordinates to the original image voxel coordinates
    # The full affine matrix is the product of the inveres of the original affine matrix, the transform affine matrix and the target affine matrix.
    # inv(original affine): maps world coordinates to voxel coordinates of the original image
    # transform affine: maps world coordinates to (transformed) world coordinates (rotation/zoom/translation)
    # target affine: maps target voxel coordinates to world coordinates
    #
    # Map: target voxel → world → transformed world → source voxel (original img)
    # => we need this in that order for scipy affine_transform function => backward mapping
    # so we map the target voxel coordinates to the original image voxel coordinates
    # where in the original image should I sample from?
    # ------------------------------------------------------------------------------------------------------
    full_affine = np.linalg.inv(original_affine) @ transform_affine @ target_affine

    # Split for scipy
    matrix = full_affine[:3, :3] #rotation/zoom matrix
    offset = full_affine[:3, 3] #translation vector

    # resample with scipy affine_transform
    resampled = affine_transform(
        input=data,
        matrix=matrix, # rotation/zoom matrix
        offset=offset, # translation vector
        output_shape=target_shape,
        order=3,  # cubic interpolation
        mode='constant', # all values outside the image are set to cval
        cval=0.0 # fill value outside the image = 0
    )

    return resampled

#%%
def build_affine(deg_x=0, deg_y=0, deg_z=0, zoom=1.0, translation_mm=(0, 0, 0)):
    """
    Build a 4x4 affine transformation matrix for rotation, zoom, and translation.
    Args:
        deg_x (float): Rotation angle x-axis.
        deg_y (float): Rotation angle y-axis.
        deg_z (float): Rotation angle z-axis.
        zoom (float): Zoom factor.
        translation_mm (tuple): Translation vector in mm.
    Returns:
        np.ndarray: 4x4 affine transformation matrix.
    """

    # convert degrees to radians
    angle_x = np.deg2rad(deg_x)
    angle_y = np.deg2rad(deg_y)
    angle_z = np.deg2rad(deg_z)

    
    rotation_x = np.eye(4)
    rotation_y = np.eye(4)
    rotation_z = np.eye(4)

    # build rotation matrix around x-axis
    rotation_x[:3, :3] = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x),  np.cos(angle_x)],
    ])

    # build rotation matrix around y-axis
    rotation_y[:3, :3] = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)],
    ])

    # build rotation matrix around z-axis
    rotation_z[:3, :3] = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z),  np.cos(angle_z), 0],
        [0, 0, 1],
    ])

    # build zoom matrix
    zoom_mat = np.diag([zoom, zoom, zoom, 1])

    # build translation matrix
    trans = np.eye(4)
    trans[:3, 3] = translation_mm

    # combine all transformations
    return trans @ zoom_mat @ rotation_x @ rotation_y @ rotation_z

#%%
class IXIDataset(Dataset):
    """IXI MRI T1 Dataset"""

    def __init__(self, img_dir, metadata_path, label_type, mask_dir=None, voxel_space = (1.0,1.0,1.0), target_shape=(256, 256, 256), transform=None):
        """"
        Args:
            img_dir (str): Folder containing .nii.gz T1 images.
            metadata_path (str): Path to IXI.xls file.
            voxel_space (tuple): Target voxel spacing in mm.
            target_shape (tuple): Output shape of each image (after resampling).
        """

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.label_type = label_type
        self.voxel_space = voxel_space
        self.target_shape = target_shape
        self.transform = transform

        # load metadata
        self.metadata = pd.read_excel(metadata_path)

        # Collect image paths
        all_img_paths = [
            os.path.join(img_dir, fname)
            for fname in os.listdir(img_dir)
            if fname.endswith(".nii.gz")
        ]

        # Filter image paths to only those with existing IXI_IDs in the metadata
        valid_img_paths = []
        valid_subject_ids = []

        # Extract available IXI_IDs from metadata
        available_ids_in_meta = set(self.metadata["IXI_ID"].astype(int))

        # Loop through all image paths and check if the subject ID is in the metadata
        for path in all_img_paths:
            match = re.search(r"IXI(\d+)-", os.path.basename(path))
            if match:
                subject_id = int(match.group(1))
                if subject_id in available_ids_in_meta:
                    row = self.metadata[self.metadata["IXI_ID"] == subject_id]
                    if not row.empty and self._has_valid_label(row.iloc[0]):
                        valid_img_paths.append(path)
                        valid_subject_ids.append(subject_id)

        self.img_paths = valid_img_paths
        self.subject_ids = valid_subject_ids

    def __len__(self):
        return len(self.img_paths)
    
    def _has_valid_label(self, row):
        if self.label_type == "sex":
            return pd.notnull(row["SEX_ID (1=m, 2=f)"])
        elif self.label_type == "age":
            return pd.notnull(row["AGE"])
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")
    
    def sample_affine(self):
        transform_config = self.transform
        if transform_config is None:
            return np.eye(4)
        
        # Sample random values from the specified ranges
        deg_x = np.random.uniform(*transform_config.get("rot_range", (0, 0)))
        deg_y = np.random.uniform(*transform_config.get("rot_range", (0, 0)))
        deg_z = np.random.uniform(*transform_config.get("rot_range", (0, 0)))

        zoom = np.random.uniform(*transform_config.get("zoom_range", (1.0, 1.0)))

        tx = np.random.uniform(*transform_config.get("translation_range", (0, 0)))
        ty = np.random.uniform(*transform_config.get("translation_range", (0, 0)))
        tz = np.random.uniform(*transform_config.get("translation_range", (0, 0)))
        translation_mm = (tx, ty, tz)

        return build_affine(deg_x, deg_y, deg_z, zoom, translation_mm)

    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = None
        if self.mask_dir is not None:
            basename = os.path.basename(img_path).replace(".nii.gz", "_mask.nii.gz")
            mask_path = os.path.join(self.mask_dir, basename)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found for {basename}: {mask_path}")
        

        subject_id = self.subject_ids[idx]


        # Lookup metadata row
        row = self.metadata[self.metadata["IXI_ID"] == subject_id]
        if row.empty:
            raise ValueError(f"No metadata found for IXI_ID: {subject_id}")
        row = row.iloc[0]
        assert self._has_valid_label(row), f"Missing label for subject {subject_id}"

        # Sample affine transformation
        transform_affine = self.sample_affine()
            
        # Load and resample image
        img_np = interpolate_to_grid(
            img_path=img_path,
            mask_path=mask_path,
            voxel_space=self.voxel_space,
            target_shape=self.target_shape,
            transform_affine=transform_affine
        )

        # Convert to tensor and add channel dimension => [C, D, H, W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

        if self.label_type == "sex":
            label = torch.tensor(int(row["SEX_ID (1=m, 2=f)"]) - 1, dtype=torch.long)
        elif self.label_type == "age":
            label = torch.tensor(row["AGE"], dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported label_type: {self.label_type}")


        return (img_tensor, label)
