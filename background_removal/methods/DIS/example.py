import os
import time
import numpy as np
import time
from glob import glob
from tqdm import tqdm
from PIL import Image

import torch, gc
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from background_removal.DIS.ISNet.models import ISNetDIS
background_image = None
image_path = "demo_datasets/finesse_model_images/finesse_6724147019853_1079_540.png"

image_name = image_path.split("/")[-1]
output_image_path = "output/DIS/finesse_model"
# background_image_path = "demo_datasets/gai_images/MODEL-ECOMM-TEMPLATE.jpg"
# background_image = Image.open(background_image_path)

model_path = "/data/segmentation/DIS/isnet-general-use.pth"
raw_image = Image.open(image_path).convert("RGB")
if background_image:
    background_image = background_image.resize(raw_image.size)
else:
    print(f"We're not going to paste the segmented image on a background as background_image_path is None")

input_image = np.asarray(raw_image)
input_size = [input_image.shape[0], input_image.shape[1]]

model = ISNetDIS()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

with torch.no_grad():
    im_shp = input_image.shape[0:2]
    im_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2,0,1).to(device)
    im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5,0.5,0.5], [1.0,1.0,1.0])
    
    result = model(image)
 
    result = torch.squeeze(F.upsample(result[0][0],im_shp,mode='bilinear'),0)
    max = torch.max(result)
    min = torch.min(result)
    result = (result - min)/(max - min)
    numpy_image = (result.squeeze().cpu().data.numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(numpy_image)

    if background_image:
        segmented_image = Image.composite(raw_image, background_image, mask_image)
    else:
        segmented_image = mask_image
    im_path = f"{output_image_path}/{image_name}"
    segmented_image.save(im_path)
    print(f"Image {image_name} segmentation using DIS finished")
