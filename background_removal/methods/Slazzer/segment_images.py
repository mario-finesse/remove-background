import os
import requests

URL = "https://api.slazzer.com/v2.0/remove_image_background"
API_KEY = "c91d2007a0624258aaa46468753afc98"

def segment_image(image_path:str, output_folder_path:str) -> None:
    image_name = os.path.basename(image_path)
    assert image_name.endswith(".png"), f"Image {image_path} must end with PNG"
    image_file = {'source_image_file': open(image_path, 'rb')}
    headers = {'API-KEY': API_KEY}
    response = requests.post(URL, files=image_file, headers=headers)
    if type(response.content) == dict:
        raise ValueError(f"{response.content}")

    output_path = f"{output_folder_path}/{image_name}"
    with open(output_path, 'wb') as img:
        img.write(response.content)

    print(f"Image {image_name} segmented")

if __name__ == "__main__":
    images_path = "data/input/finesse_model_images"
    output_folder_path = "data/output/Slazzer"
    png_files = [file for file in os.listdir(images_path) if file.endswith('.png')]
    for i, image_name in enumerate(png_files):
        if image_name not in os.listdir(output_folder_path):
            print(f"{i}/{len(png_files)}: Segmenting image {images_path}")
            segment_image(os.path.join(images_path, image_name), output_folder_path)
        else:
            print(f"{i}/{len(png_files)}: Image {image_name} already exists in {output_folder_path}")

