from typing import Union, List
from sqlalchemy.orm.query import Query
from sqlalchemy.orm import joinedload

from product_dataset.models import ProductData, ModelGrayImage, MidjourneyGeneratedImage
from product_dataset.services import ProductsDB, AdvancedProductsDB, MidjourneyProductsDB
from db_manager import db_operation, db

@db_operation
def get_products_image_desc_pair() -> List[ProductData]:
    query = db.session.query(ProductData)
    query = query.options(joinedload(ProductData.model_gray_image)) \
                .join(ProductData.model_gray_image) \
                .filter(ProductData.description.isnot(None)) \
                .filter(ModelGrayImage.front.isnot(None))
    return query.all()