from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import os
import torch
import torchio as tio
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Using TorchIO images without a torchio\.SubjectsLoader.*",
    category=UserWarning
)

def get_sub_volume(img_path, label_l_path, label_r_path, output_x=32, output_y=32, output_z=32, label_prob=0.9, train_split=True, pad=0):

    img = nib.load(img_path)
    data = img.get_fdata()

    label_l = nib.load(label_l_path)
    label_l_data = label_l.get_fdata()

    label_r = nib.load(label_r_path)
    label_r_data = label_r.get_fdata()

    # Combine left and right labels
    label_data = (label_l_data * 2) + label_r_data

    # check that its not overlapping
    if label_data.max() > 2:
        raise ValueError("Labels are overlapping.")
    
    patch_img = None
    patch_label = None
    
    if train_split:
        orig_x, orig_y, orig_z = data.shape

        # creat a random number to select the patch
        prob = np.random.rand()

        # check if the random number is less than the label probability
        if prob < label_prob:
            # select the indices of the label data where the label is greater than 0 (not background)
            indices = np.argwhere(label_data > 0)

            # randomly select a indices of a point on the label and use it as the center of the patch
            center_x, center_y, center_z = indices[np.random.randint(len(indices))]
        else:
            # Randomly select a center point for the patch
            center_x = np.random.randint((output_x // 2), orig_x - (output_x//2) +1)
            center_y = np.random.randint((output_y//2), orig_y - (output_y//2) +1)
            center_z = np.random.randint((output_z//2), orig_z - (output_z//2) +1)

        
        # Calculate the start indices for the patch & ensure they are within bounds => clip the values to be not less than 0 and not greater than the original shape - output shape
        start_x = np.clip(center_x - (output_x // 2), 0, orig_x - output_x)
        start_y = np.clip(center_y - (output_y // 2), 0, orig_y - output_y)
        start_z = np.clip(center_z - (output_z // 2), 0, orig_z - output_z)

        # select the patch
        patch_img = data[start_x:start_x + output_x,
                            start_y:start_y + output_y,
                            start_z:start_z + output_z]
        
        patch_label = label_data[start_x:start_x + output_x,
                                    start_y:start_y + output_y,
                                    start_z:start_z + output_z]
    else:
        patch_img = np.pad(data, ((pad,pad), (pad, pad), (pad, pad)), mode='reflect')
        patch_label = np.pad(label_data, ((pad,pad), (pad, pad), (pad, pad)), mode='reflect')

    return patch_img, patch_label

class ADNIDataset(Dataset):
    def __init__(self, folder_path, split="train", patch_size=(32, 32, 32), label_prob=0.9, transform=None):
        
        if split not in ["train", "test"]:
            raise ValueError("split must be either 'train' or 'test'")

        self.folder_path = folder_path
        self.train_split = split == "train"
        self.patch_size = patch_size
        self.label_prob = label_prob
        self.transform = transform

        self.label_folder = None
        self.img_folder = os.path.join(folder_path, "ADNI")

        self.full_paths = []

        # check which folder to use
        if self.train_split:
            self.label_folder = os.path.join(folder_path, "Labels_100_NIFTI")
        else:
            self.label_folder = os.path.join(folder_path, "Labels_35_NIFTI")

        # Prepare list of samples
        self.full_paths = self.create_samples_list()

    def create_samples_list(self):
        subject_ids = []

        # loop through the label folder and get all the subject_ids
        for file in os.listdir(self.label_folder):
            
            # check if the file is a nii.gz file and contains L or R hyppocampus label
            if file.endswith(".nii.gz") and ("_L" in file or "_R" in file) and "CSF" not in file:
                
                # Extract subject_id from the filename
                file_parts = file.split("_")
                subject_ids.append("_".join(file_parts[1:5]))

        # Remove duplicates by converting to a set
        unique_subject_ids = set(subject_ids)

        # check if we have 2 labels for each subject
        if len(subject_ids)/len(unique_subject_ids) != 2:
            raise ValueError("There are missing labels for some subjects, please check the label folder.")

        paths = []

        # loop through the unique subject_ids
        for subject_id in unique_subject_ids:
            # Construct the paths for the left and right labels
            label_l_path = os.path.join(self.label_folder, f"ADNI_{subject_id}_L.nii.gz")
            label_r_path = os.path.join(self.label_folder, f"ADNI_{subject_id}_R.nii.gz")

            # creat the path to the first subfolder for the image
            img_sub_folder = subject_id.split("_")
            img_sub_folder = "_".join(img_sub_folder[:3])

            img_path = os.path.join(self.img_folder, img_sub_folder)

            found_nii_files = []

            # walk through the image folder and get all the nii.gz files => should be only one
            for root, _, files in os.walk(img_path):
                nii_files = [f for f in files if f.endswith(".nii.gz")]
                if len(nii_files) > 0:
                    # Append full paths of all nii files found
                    found_nii_files.extend([os.path.join(root, f) for f in nii_files])

            # check if we have only one nii file
            if len(found_nii_files) == 1:
                img_path = found_nii_files[0]

                # Load the image and compute mean & std
                img_nib = nib.load(img_path)
                img_data = img_nib.get_fdata().astype(np.float32)

                mean = img_data.mean()
                std = img_data.std()

                if std == 0:
                    std = 1.0

            elif len(found_nii_files) == 0:
                raise ValueError(f"No scan found for subject {subject_id}, manual check needed.")
            else:
                raise ValueError(f"Multiple scans found for subject {subject_id}, manual check needed.")

            # check if the paths exist
            if os.path.exists(label_l_path) and os.path.exists(label_r_path) and os.path.exists(img_path):
                # Append the paths to the list
                paths.append({
                    "img": img_path,
                    "label_l": label_l_path,
                    "label_r": label_r_path,
                    "subject_id": subject_id,
                    "mean": mean,
                    "std": std
                })

        return paths
    

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx):

        patch_img, patch_label = get_sub_volume(
            self.full_paths[idx]["img"],
            self.full_paths[idx]["label_l"],
            self.full_paths[idx]["label_r"],
            output_x=self.patch_size[0],
            output_y=self.patch_size[1],
            output_z=self.patch_size[2],
            label_prob=self.label_prob,
            train_split=self.train_split,
            pad=44  # Padding to ensure the receptive field is covered
        )

        # Ensure channel dimension: [1, D, H, W]
        patch_img = np.expand_dims(patch_img, axis=0)

        # Normalize the image
        patch_img = (patch_img - self.full_paths[idx]["mean"]) / self.full_paths[idx]["std"]

        # Convert to tensors
        patch_img = torch.from_numpy(patch_img).float()
        patch_label = torch.from_numpy(patch_label).long()

        if self.transform is not None:
            subject = tio.Subject(
                img=tio.ScalarImage(tensor=patch_img),
                label=tio.LabelMap(tensor=patch_label.unsqueeze(0))  # torchio expects [C, D, H, W]
            )
            transformed = self.transform(subject)
            patch_img = transformed['img'].data  # [1, D, H, W]
            patch_label = transformed['label'].data.squeeze(0).long()  # back to [D, H, W]

        return patch_img, patch_label, self.full_paths[idx]["mean"], self.full_paths[idx]["std"]



        

