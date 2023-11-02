import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from background_removal.methods.DIS.ISNet.data_loader_cache import normalize, im_reader, im_preprocess
from background_removal.methods.DIS.ISNet.models import *


class GOSNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image_size = image.size
        image = normalize(image, self.mean, self.std)
        return image

class BackgroundRemoval:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose(
            [GOSNormalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])])
        # Set Parameters
        self.hypar = {}  # paramters for inferencing
        self.hypar["model_path"] = os.getcwd() + '/saved_models/IS-Net/'
        # name of the to-be-loaded weights
        self.hypar["restore_model"] = "isnet-general-use.pth"
        # indicate if activate intermediate feature supervision
        self.hypar["interm_sup"] = False
        # choose floating point accuracy --
        # indicates "half" or "full" accuracy of float number
        self.hypar["model_digit"] = "full"
        self.hypar["seed"] = 0
        # cached input spatial resolution, can be configured into different size
        self.hypar["cache_size"] = [1024, 1024]
        # data augmentation parameters ---
        # mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
        self.hypar["input_size"] = [1024, 1024]
        # random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation
        self.hypar["crop_size"] = [1024, 1024]
        self.hypar["model"] = ISNetDIS()

        # Build Model
        self.net = self.build_model()
    
    def load_image(self, im: Image):
        if im.mode == 'RGBA':
            im = im.convert('RGB')
        im = np.array(im)
        im, im_shp = im_preprocess(im, self.hypar["cache_size"])
        im = torch.divide(im, 255.0)
        shape = torch.from_numpy(np.array(im_shp))
        # make a batch of image, shape
        return self.transform(im).unsqueeze(0), shape.unsqueeze(0)

    def build_model(self):
        self.net = self.hypar["model"]  # GOSNETINC(3,1)
        # convert to half precision
        if (self.hypar["model_digit"] == "half"):
            self.net.half()
            for layer in self.net.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()
        self.net.to(self.device)
        if (self.hypar["restore_model"] != ""):
            self.net.load_state_dict(torch.load(
                self.hypar["model_path"]+"/"+self.hypar["restore_model"], map_location=self.device))
            self.net.to(self.device)
        self.net.eval()
        return self.net

    def predict_mask(self,  inputs_val, shapes_val):    
        self.net.eval()
        if (self.hypar["model_digit"] == "full"):
            inputs_val = inputs_val.type(torch.FloatTensor)
        else:
            inputs_val = inputs_val.type(torch.HalfTensor)
        inputs_val_v = Variable(inputs_val, requires_grad=False).to(
            self.device)  # wrap inputs in Variable
        ds_val = self.net(inputs_val_v)[0]  # list of 6 results
        # B x 1 x H x W    # we want the first one which is the most accurate prediction
        pred_val = ds_val[0][0, :, :, :]

        # recover the prediction spatial size to the orignal image size
        # pred_val = torch.squeeze(F.upsample(torch.unsqueeze(
        #     pred_val, 0), (shapes_val[0][0], shapes_val[0][1]), mode='bilinear'))
        pred_val = torch.squeeze(F.interpolate(torch.unsqueeze(
            pred_val, 0), size=(shapes_val[0][0], shapes_val[0][1]), mode='bilinear', align_corners=False))


        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val-mi)/(ma-mi)  # max = 1

        if self.device == 'cuda':
            torch.cuda.empty_cache()
        # it is the mask we need
        return (pred_val.detach().cpu().numpy()*255).astype(np.uint8)

    def remove_background(self, image: Image) -> Image:
        image_path = image
        image_tensor, orig_size = self.load_image(image_path)
        mask = self.predict_mask(image_tensor, orig_size)

        pil_mask = Image.fromarray(mask).convert('L')
        im_rgb = image.convert("RGB")

        im_rgba = im_rgb.copy()
        im_rgba.putalpha(pil_mask)
        return im_rgba

if __name__ == "__main__":
    current_dir = os.getcwd()
    image_path= current_dir + "/demo_datasets/shopify_img/shopify_img_4.png"
    im = Image.open(image_path)
    background_removal = BackgroundRemoval()
    imrgba =  background_removal.remove_background(im)
    imrgba.save(current_dir + "/demo_datasets/shopify_img_alpha/shopify_img_4.png")