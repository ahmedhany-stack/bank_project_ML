import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

engine = None

try:
    # الاتصال بـ MySQL
    engine = create_engine(
        "mysql+pymysql://root:Ataazee66@127.0.0.1/bank_db"
    )

    query = "SELECT * FROM bank_marketing"

    df = pd.read_sql(query, engine)

    logging.info(f"Successfully loaded {len(df)} rows")

    print(df.head())

except Exception as e:
    logging.error(f"Failed to read data from MySQL: {e}")

finally:
    if engine is not None:
        engine.dispose()
        logging.info("Database connection closed")