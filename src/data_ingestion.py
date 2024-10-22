import pandas as pd
import os
import yaml
import logging
import requests




def main():
    logger = logging.getLogger("DataIngestion")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        with open('params.yaml', 'r') as ymlfile:
            params = yaml.safe_load(ymlfile)
            logger.info("Params loaded")

        url = params['data_ingestion']['url']
        file_path = os.path.join("data","raw", "reddit.csv")

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
            df.to_csv(file_path, index=False)
            logger.info("Data saved")

        except FileNotFoundError as exc:
            logger.error(f"file save_path {file_path} not found")
            raise
        except Exception as exc:
            logger.error(exc)
            raise

    else:
        logger.error(f"Failed to fetch data due to URLError{request.status_code}")
        raise requests.exceptions.HTTPError

if __name__ == "__main__":
    main()