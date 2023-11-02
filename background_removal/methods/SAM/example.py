import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from background_removal.SAM.show_masks import *

device = "cuda" if torch.cuda.is_available() else "cpu"
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

image_path = "demo_datasets/gai_images/013.png"
raw_image = Image.open(image_path).convert("RGB")

input_points = [[[448, 576]]]  # 2D location of the middle of the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = sam_model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

show_masks_on_image(raw_image, masks[0], scores)