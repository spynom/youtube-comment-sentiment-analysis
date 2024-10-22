import pandas as pd
import os
import yaml
import logging
import requests

# Set up logging for the data ingestion process
logger = logging.getLogger("DataIngestion")
logger.setLevel(logging.DEBUG)

# Create a stream handler to output logs to the console
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

# Define the logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_yaml(path: str) -> dict:
    """Read parameters from a YAML file."""
    try:
        with open(path, 'r') as ymlfile:
            # Load data ingestion parameters from the YAML file
            params = yaml.safe_load(ymlfile)['data_ingestion']
            logger.info("Params YAML file read: %s", path)
            return params

    except FileNotFoundError as e:
        logger.error(f"Params.yaml file not found: {e}")
        raise

    except Exception as e:
        logger.error(f"Error reading params.yaml: {e}")
        raise


def fetch_data(url: str) -> pd.DataFrame:
    """Fetch data from a given URL and return it as a DataFrame."""
    # Send a GET request to the URL
    request = requests.get(url)

    if request.status_code == 200:
        # Read the data into a DataFrame if the request was successful
        df = pd.read_csv(url)
        logger.info("Data fetched successfully")
        return df
    else:
        logger.error(f"Failed to fetch data due to URL error: {request.status_code}")
        raise requests.exceptions.HTTPError


def save_data(df: pd.DataFrame, file_path: str):
    """Save the DataFrame to a CSV file."""
    try:
        # Save the DataFrame as a CSV file
        df.to_csv(file_path, index=False)
        logger.info("Data saved to %s", file_path)

    except Exception as exc:
        logger.error(exc)
        raise


def main():
    """Main function to orchestrate data ingestion."""
    # Read parameters from the YAML file
    params = read_yaml("params.yaml")

    # Define the path to save the raw data
    file_path = os.path.join("data", "raw", "reddit.csv")

    # Fetch data using the URL from parameters and save it to the specified path
    data = fetch_data(params['url'])
    save_data(data, file_path)


# Entry point for the script
if __name__ == "__main__":
    main()
