from airflow.sdk import dag,task
from datetime import datetime
from src.main import main

@dag(
    dag_id="ML_dag",
    schedule="@daily",
    start_date=datetime(2026, 6,25),
    catchup=False,
    tags=["ML_pipeline"],
)
def first_dag():
    @task()
    def run_pipeline():
        
        main()

    run_pipeline()

first_dag()
