from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from jobspy import scrape_jobs
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath("/home/jiaqiliu/src/"))
# Importing your custom modules
from data_fetching import update_job_data
from data_cleaning import clean_data
from jd_embeddings import compute_job_embeddings

# Base directory for storing output files
BASE_DIR = "/home/jiaqiliu/data/"
os.makedirs(BASE_DIR, exist_ok=True)  # Ensure the directory exists

default_args = {
    'owner': 'jiaqi',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'job_data_pipeline_new',
    default_args=default_args,
    description='Pipeline for fetching and processing job data',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['job_data'],
)

# Define the tasks
def fetch_job_data(**context):
    """Task to fetch job data"""
    from jobspy import scrape_jobs
    import pandas as pd
    import os
    import sys
    sys.path.append(os.path.abspath("/home/jiaqiliu/src/"))
    BASE_DIR = "/home/jiaqiliu/data/"
    os.makedirs(BASE_DIR, exist_ok=True)  # Ensure the directory exists
    from data_fetching import update_job_data
    try:
        output_file = os.path.join(BASE_DIR, "job_data.csv")
        job_titles = [
            "software engineer",
            "data scientist",
            "data analyst",
            "sales",
            "human resources"
        ]
        update_job_data(
            job_titles=job_titles,
            location="United States",
            hours_old=24,
            results_per_query=30,
            total_results_wanted=500,
            output_file=output_file
        )
        print(f"Data saved to {output_file}")
    except Exception as e:
        print(f"Error in fetch_job_data: {e}")
        raise

def process_job_data(**context):
    """Task to clean the job data"""
 #   input_file = context['task_instance'].xcom_pull(task_ids='fetch_job_data')  # Get file path from XCom
    input_file = os.path.join(BASE_DIR, "job_data.csv")
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    df = pd.read_csv(input_file)
    cleaned_df = clean_data(df)

    output_file = os.path.join(BASE_DIR, "processed_job_data.csv")
    cleaned_df.to_csv(output_file, index=False)

    return output_file  # Pass file path to XCom

def generate_embeddings(**context):
    """Task to generate embeddings for the cleaned job data"""
    input_file = context['task_instance'].xcom_pull(task_ids='process_job_data')  # Get file path from XCom
    if not input_file or not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found")

    df = pd.read_csv(input_file)

    compute_job_embeddings(df)

start = EmptyOperator(
    task_id='start',
    dag=dag,
)

# fetch_data = PythonOperator(
#     task_id='fetch_job_data',
#     python_callable=fetch_job_data,
#     dags=dags,
# )
fetch_data = PythonVirtualenvOperator(
    task_id='fetch_job_data',
    python_callable=fetch_job_data,
    requirements=["python-jobspy", "pandas"],
    system_site_packages=False,
    dag=dag,
)


process_data = PythonOperator(
    task_id='process_job_data',
    python_callable=process_job_data,
    dag=dag,
)

generate_embeddings_task = PythonOperator(
    task_id='generate_embeddings',
    python_callable=generate_embeddings,
    dag=dag,
)

end = EmptyOperator(
    task_id='end',
    dag=dag,
)

start >> fetch_data >> process_data >> generate_embeddings_task >> end
