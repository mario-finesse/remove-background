import pandas as pd
import wandb
import PIL
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import csv

from product_dataset.main import ProductDataset
from storage_utils import S3Helper, DataframeHelper
from background_removal import BackgroundRemoval

class VisualizeBackgroundRemoval:
    def __init__(self, wandb:bool = False):
        self.wandb = wandb
        self.back_remove = BackgroundRemoval()

    def visualise_shopify_3d_renders(self, res:int = 1024,
                                     filename:str = 'demo_datasets/products-shopify-3d-renders.csv'):
        if self.wandb:
            wandb.init(project="background-removal",
                       name="shopify-vs-white-vs-3d-render",)
        prods_list = []
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)        
            for row in csvreader:
                for number in row:
                    if number.strip():  # Check if the value is not empty
                        prods_list.append(int(number))
        
        products = ProductDataset(data_root="/data/.cache/", force_download=True).products
        df = pd.DataFrame(columns=['Product ID'])
        i = 0
        for product_id in tqdm(prods_list, desc="Shopify vs. 3d renders"):
            image_name = products.loc[product_id, f'Image {res} white']
            image = S3Helper().load_image(s3_folder=f'front-white-{res}-images', filename=image_name)
            shopify_img = None
            image_url = None
            if products.loc[product_id, 'Shopify Image']:
                image_url = products.loc[product_id, 'Shopify Image']
                if "1024x1024" in image_url:
                    image_url = image_url.replace("1024x1024", "2000x2000")
                response = requests.get(image_url)
                try:
                    shopify_img = Image.open(BytesIO(response.content))
                    shopify_img_noback = self.back_remove.remove_background(shopify_img)
                    background = Image.new('RGBA', shopify_img_noback.size, (255, 255, 255, 255))
                    background.paste(shopify_img_noback, mask=shopify_img_noback)
                    shopify_img = wandb.Image(shopify_img)
                    background = wandb.Image(background)
                except PIL.UnidentifiedImageError:
                    shopify_img = None
                    background = None
                except OSError:
                    shopify_img = None
                    background = None
                
            df = df.append({
                'Product ID': product_id,
                'Title': products.loc[product_id, 'Title'],
                'Drop': products.loc[product_id, 'Drop'],
                'Description': products.loc[product_id, 'Description'],
                'Shopify URL': image_url,
                'Shopify Image': shopify_img,
                'Shopify white background': background,
                '3D rendered Image': wandb.Image(image),
            }, ignore_index=True)
            i = i + 1
            if i > 200:
                break
        if self.wandb:
            wandb.log({"shopify-whiteshopify-white": wandb.Table(dataframe=df)})

    def visualise_no_img(self):
        if self.wandb:
            wandb.init(project="background-removal",
                       name="no-img",)
        filename = 'demo_datasets/products-no-img.csv'
        no_img_df = DataframeHelper().load_dataframe(filename='products-no-img.csv', folder='demo_datasets', idx=True)
        products = ProductDataset(data_root="/data/.cache/", force_download=False).products
        df = pd.DataFrame(columns=['Product ID'])
        i = 0
        for product_id, _ in tqdm(no_img_df.iterrows(), total=no_img_df.shape[0], desc="Getting images for no-img"):
            shopify_img = None
            image_url = None
            if products.loc[product_id, 'Shopify Image']:
                image_url = products.loc[product_id, 'Shopify Image']
                if "1024x1024" in image_url:
                    image_url = image_url.replace("1024x1024", "2000x2000")
                response = requests.get(image_url)
                try:
                    shopify_img = Image.open(BytesIO(response.content))
                    shopify_img_noback = self.back_remove.remove_background(shopify_img)
                    # background = Image.new('RGBA', shopify_img_noback.size, (255, 255, 255, 255))
                    # background.paste(shopify_img_noback, mask=shopify_img_noback)
                    shopify_img = wandb.Image(shopify_img)
                    shopify_img_noback = wandb.Image(shopify_img_noback)
                except PIL.UnidentifiedImageError:
                    shopify_img = None
                    shopify_img_noback = None
                except OSError:
                    shopify_img = None
                    shopify_img_noback = None
                
            df = df.append({
                'Product ID': product_id,
                'Title': products.loc[product_id, 'Title'],
                'Drop': products.loc[product_id, 'Drop'],
                'Description': products.loc[product_id, 'Description'],
                'Shopify URL': image_url,
                'Shopify Image': shopify_img,
                'No back Image': shopify_img_noback,
            }, ignore_index=True)
            i = i + 1
            if i > 300:
                break
        if self.wandb:
            wandb.log({"shopify-noback-white": wandb.Table(dataframe=df)})

if __name__ == '__main__':
    vis = VisualizeBackgroundRemoval(wandb=True)
    vis.visualise_shopify_3d_renders(filename='demo_datasets/failed-back-removal.csv')