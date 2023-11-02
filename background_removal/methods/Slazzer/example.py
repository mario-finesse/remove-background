import requests
import os

URL = "https://api.slazzer.com/v2.0/remove_image_background"
API_KEY = "c91d2007a0624258aaa46468753afc98"
image_path = "data/input/finesse_model_images/finesse_6661712707661_1387_694.png"
image_name = os.path.basename(image_path)
output_path = "data/output/Slazzer"


image_file = {'source_image_file': open(image_path, 'rb')}
headers = {'API-KEY': API_KEY}
response = requests.post(URL, files=image_file, headers=headers)

output_path = f"{output_path}/{image_name}"
with open(output_path, 'wb') as img:
    img.write(response.content)

print(f"Image {image_name} segmented")