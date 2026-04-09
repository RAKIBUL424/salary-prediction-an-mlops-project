import os
import sys
import yaml
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split


def load_yaml(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
            logging.info(f"Successfully loaded YAML file: {file_path}")
            return params
    except Exception as e:
        logging.error(f"Error loading YAML file: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from: {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        raw = os.path.join(data_path, "raw")
        os.makedirs(raw, exist_ok=True)
        train_data.to_csv(os.path.join(raw, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw, "test.csv"), index=False)
        logging.debug(f"Data saved successfully at {data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_yaml("params.yaml")
        test_size = params['data_ingestion']['test_size']
        random_state = params['data_ingestion']['random_state']
        data_url = params['data_ingestion']['data_url']
        data_output_path = params['data_ingestion']['data_output_path']

        df = load_data(data_url)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        save_data(train_data, test_data, data_output_path)
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.error(f"Data ingestion failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()