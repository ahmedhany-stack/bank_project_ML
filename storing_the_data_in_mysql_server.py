import pandas as pd
from sqlalchemy import create_engine
import logging


import os
logging.basicConfig(level=logging.DEBUG,filename="log.log",filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

logger=logging.getLogger(__name__)
handler=logging.FileHandler("log.log")
formatter=logging.Formatter(f"%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

file_path = r"C:\Users\AS\Downloads\archive(4)\bank-full.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError("CSV file not found")
    

# load data
def  main():
    df = pd.read_csv(file_path)
    if df.empty:
      raise ValueError("DataFrame is empty")
    if df["age"].min() < 0:
      raise ValueError("Invalid age detected")
    logger.info(f"Rows: {len(df)}")
    logger.info(f"Duplicates: {df.duplicated().sum()}")
    logger.info(f"Null values:\n{df.isnull().sum()}")
    before = len(df)
    df.drop_duplicates(inplace=True)

    
    after = len(df)

    logger.info(f"Removed {before - after} duplicate rows")
    expected_columns = 17

    if len(df.columns) != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns but found {len(df.columns)}"
        )


    # rename columns لو لازم (اختياري حسب اسم الملف عندك)
    df.columns = [
        "age","job","marital","education","default_flag",
        "balance","housing","loan","contact","day",
        "month","duration","campaign","pdays",
        "previous","poutcome","deposit"
    ]


    # connect mysql
    engine = create_engine(
        "mysql+pymysql://root:Ataazee66@127.0.0.1/bank_db",
        pool_pre_ping=True
    )



    logger.info("Starting data load")

    try:
        df.to_sql(
            name="bank_marketing",
            con=engine,
            if_exists="append",
            index=False,
            chunksize=1000,
            method="multi"
        )
        logger.info("data load completed")
    except Exception as e:
        logger.error("Error occurred while loading data to MySQL: %s", e)
    finally:
        engine.dispose()
        logger.info("Database connection closed")
        


if __name__ == "__main__":
    main()

