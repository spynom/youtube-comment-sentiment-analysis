import pandas as pd
import os
import yaml
import logging
import requests



def fetch_dataset():
    try:
        with open('params.yaml', 'r') as ymlfile:
            params = yaml.safe_load(ymlfile)

        url = params['data_ingestion']['url']
        save_dir = params['data_ingestion']['save_dir']
        file_path = os.path.join(save_dir,"reddit.csv")

    except FileNotFoundError as exc:
        logger.error(exc)
        raise

    except yaml.YAMLError as exc:
        logger.error(exc)
        raise
    except Exception as exc:
        logger.error(exc)
        raise

    request = requests.get(url)

    if request.status_code == 200:
        df = pd.read_csv(url)
        try:
            df.to_csv(file_path,index=False)

        except FileNotFoundError as exc:
            logger.error(f"file save_path {save_dir} not found")
        except Exception as exc:
            logger.error(exc)

    else:
        logger.error(f"Failed to fetch data due to URLError{request.status_code}")
        raise









if __name__ == "__main__":
    logger = logging.getLogger("DataIngestion")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    fetch_dataset()