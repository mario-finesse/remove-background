import requests
from background_removal.database import setup, get

API_KEY = "JB6oj4xOjtMPpwkhy7RBcoNTFsCWvyFs"
SECRET_KEY = "QGU8xxDcT01YpOe3"

products_service = setup.setup_db_products(database_name="finesse_products")
products = get.get_products_image_desc_pair()

for product in products:
    image_url = product.model_gray_image.front
    print(f"Segmenting image: {image_url}")
    url = "https://api.picsart.io/tools/1.0/removebg"
    payload={
            "format": "PNG",
            "output_type": "cutout",
            "image_url": image_url
            }
    headers = {
      "accept": "application/json",
      "x-picsart-api-key": API_KEY 
    }

    response = requests.request(method="POST", url=url, data=payload, headers=headers)

    print(response.text)
    break