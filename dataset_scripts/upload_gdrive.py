import os
from storage_utils import GoogleDriveAPI

gdrive = GoogleDriveAPI()
images_path = "data/image_background/10d72c"
png_files = [file for file in os.listdir(images_path) if file.endswith('.png')]
for i, image_name in enumerate(png_files):
    source_file_path = os.path.join(images_path, image_name)
    print(f"{i}/{len(png_files)}: Uploading image {source_file_path}")
    gdrive.upload_file(file_path=source_file_path, gdrive_folder_name="10d72c", new_file_name=None)