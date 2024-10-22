import pandas as pd
import os
import yaml

def fetch_dataset():
    url = yaml.load(open('params.yaml', 'r'))["data_ingestion"]["url"]
    save_path = os.path.join("data","raw")

    pd.read_csv(url).to_csv(os.path.join(save_path,"reddit.csv"),index=False)

if __name__ == "__main__":
    fetch_dataset()