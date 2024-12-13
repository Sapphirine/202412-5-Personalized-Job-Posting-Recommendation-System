# Final Project 6893 Personalized Job Posting Recommendation System

This repository contains the final project for the course EECS 6893. 

## Repository Structure

```
final_project_6893/
├── airflow-dag/
│   └── [DAG files for orchestrating workflows using Apache Airflow]
├── app/
│   └── [Source code for the application interface]
├── data/
│   └── [datasets and saved models used for the project]
├── src/
│   └── [Core Python scripts for data processing and model building]
```

## Key Features

- **Data Workflow Orchestration**
- **Recommendation Engine**: Implements a machine learning-based recommendation system.

## Prerequisites

- Python 3.8+
- Required Python libraries (listed in `requirements.txt`)
- openai API key: get it from https://platform.openai.com/
  
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/binruiyang/final_project_6893.git
   cd final_project_6893
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. unzip job_data.csv

4. Optional Set up Apache Airflow:
   - Follow [Apache Airflow's official guide](https://airflow.apache.org/docs/apache-airflow/stable/start.html) to install and configure Airflow.

## Usage

To run the application: 
```streamlit run app/app.py```



## Authors

- **Binrui Yang**
- **Jiaqi Liu**
- **Tianyi Chen**
