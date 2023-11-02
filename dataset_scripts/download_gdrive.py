import os
from storage_utils import GoogleDriveAPI

gdrive = GoogleDriveAPI()
gdrive_folder = "true-back-e5e6eb"
destination_path = "/data/clogen/ed_datasets/diff_backgrounds/true-back-e5e6eb"
print(f"Starting download from {gdrive_folder} to {destination_path}")
gdrive.download_folder(gdrive_folder_name=gdrive_folder, output_dir=destination_path)
print(f"Download Finished to {destination_path}")