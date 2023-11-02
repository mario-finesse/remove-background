import os
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import numpy as np

class SAMSegment():
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    def segment_image(self, image_path:str, output_folder_path:str, background_image_path:str) -> None:
        image_name = image_path.split("/")[-1]
        raw_image = Image.open(image_path).convert("RGB")
        background_image = Image.open(background_image_path)
        background_image = background_image.resize(raw_image.size)
        
        inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
        image_embeddings = self.sam_model.get_image_embeddings(inputs["pixel_values"])

        input_image = np.asarray(raw_image)
        input_points = [[[(input_image.shape[0])//2, (input_image.shape[1])//2]]] # 2D location of the middle of the image
        
        inputs = self.processor(raw_image, input_points=input_points, return_tensors="pt").to(self.device)

        inputs.pop("pixel_values", None)
        inputs.update({"image_embeddings": image_embeddings})

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
        masks = masks[0].squeeze()
        mask = masks[2]

        mask_image = Image.fromarray((mask.numpy() * 255).astype('uint8'))  # Convert tensor to 8-bit image
        segmented_image = Image.composite(raw_image, background_image, mask_image)

        im_path = f"{output_folder_path}/{image_name}"
        segmented_image.save(im_path)
        print(f"Image {image_name} segmentation using SAM finished")

if __name__ == "__main__":
    segment = SAMSegment()
    images_path = "demo_datasets/gai_images/sdxl-human"
    output_folder_path = "output/SAM"
    background_image_path = "demo_datasets/gai_images/MODEL-ECOMM-TEMPLATE.jpg"
    for image_name in os.listdir(images_path):
        print(f"Segmenting image {image_name}")
        segment.segment_image(os.path.join(images_path, image_name), output_folder_path, background_image_path)