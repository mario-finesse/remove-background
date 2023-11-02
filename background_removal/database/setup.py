from typing import Tuple
from product_dataset.services import AdvancedProductsDB

from db_manager import DBConfig, init_db, db_operation
from product_dataset import PRODUCTS_BIND_KEY

@db_operation
def get_products_service() -> AdvancedProductsDB:
    return AdvancedProductsDB()

def get_products_config(database_name:str="finesse_products") -> DBConfig:
    assert PRODUCTS_BIND_KEY is not None, "product_dataset/__init__.py does not contain CLV_BIND_KEY, make sure you're in the correct branch"
    assert database_name in ["finesse_products", "mj_labels"], f"database_name must be either finesse_products or mj_labels, not {database_name}"
    print(f"PRODUCTS_BIND_KEY: ", PRODUCTS_BIND_KEY)
    # CONNECTION PARAMS
    USERNAME = 'products'
    PASSWORD = 'Finesseproducts'
    SERVER_URL = 'finesse-products.cv5bmpgqjjbf.us-west-2.rds.amazonaws.com'
    PORT = '5432'
    DB_NAME = database_name
    products_config = DBConfig(USERNAME, PASSWORD, SERVER_URL, PORT, DB_NAME, PRODUCTS_BIND_KEY)
    return products_config

def setup_db_products(database_name:str="finesse_products"):
    products_config = get_products_config(database_name)
    print(f'Connecting to DB at: {products_config.url}')
    init_db(db_configs = [products_config])
    products_service = get_products_service()
    return products_service