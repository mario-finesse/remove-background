import os
images_path = "/data/clogen/ed_datasets/diff_backgrounds/true-back-e5e6eb"
background = "e5e6eb"
# assert background == images_path.split("/")[-1], f"{background} not the same as the folder specified"
png_files = [file for file in os.listdir(images_path) if file.endswith('.png')]
for i, file_name in enumerate(png_files):
    new_name = os.path.splitext(file_name)[0] + f'_{background}' + os.path.splitext(file_name)[1]
    original_path = os.path.join(images_path, file_name)
    new_path = os.path.join(images_path, new_name)
    print(f"{i}/{len(png_files)}: Renaming {original_path} to {new_path}")
    os.rename(original_path, new_path)
