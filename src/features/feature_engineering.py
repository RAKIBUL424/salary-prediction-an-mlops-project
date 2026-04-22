import os
import pandas as pd
import yaml
from src.logger import logging



def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info("Parameters loaded successfully.")
        return params
    except Exception as e:
        logging.error(f"Error occurred while loading parameters: {e}")
        raise

def load_data(data_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(data_path)
        logging.info("Data loaded successfully.")
        return data
    except Exception as e:
        logging.error(f"Error occurred while loading data: {e}")
        raise

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['experience_level'] = pd.cut(data['experience_years'], bins=[0, 2, 5, 10,50], labels=['Junior', 'Mid', 'Senior', 'Expert'])
        data['skill_density'] = data['skills_count'] / (data['experience_years'] + 1)
        data['cert_per_year'] = data['certifications'] / (data['experience_years'] + 1)
        data['total_qualifications'] = data['skills_count'] + data['certifications']
        data['exp_x_skills'] = data['experience_years']*data['skills_count']
        data['exp_x_cert'] = data['experience_years']*data['certifications']
        data['is_tech'] = (data['industry'] == 'Tech').astype(int)
        data['is_masters_plus'] = (data['education_level'].isin(["Master's", "PhD"])).astype(int)
        logging.info("Feature engineering completed successfully.")
        return data
    except Exception as e:
        logging.error(f"Error occurred during feature engineering: {e}")
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    try:
        interim = os.path.join(data_path)
        os.makedirs(interim, exist_ok=True)
        train_data.to_csv(os.path.join(interim,"train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim,"test_processed.csv"), index=False)
        logging.debug(f"Data saved successfully at {data_path}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise

def main():
    try:
        params = load_params('params.yaml')
        train_data_path = params['feature_engineering']['train_data_path']
        test_data_path = params['feature_engineering']['test_data_path']
        output_data_path = params['feature_engineering']['data_output_path']
        train_data = load_data(train_data_path)
        test_data = load_data(test_data_path)
        train_df = feature_engineering(train_data)
        test_df = feature_engineering(test_data)

        save_data(train_df, test_df, output_data_path)
        logging.info("Feature engineering process completed successfully.")
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        raise

if __name__ == "__main__":
    main()