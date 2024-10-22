import pandas as pd
from lightgbm import LGBMClassifier
import os
import yaml
import pickle
from sklearn.metrics import classification_report
import json
import logging

logger = logging.getLogger("Feature Engineering")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)



try:
    with open("params.yaml") as f:
        params = yaml.safe_load(f) ["model_building"]
    logger.info("params yaml file is read successfully")

except FileNotFoundError:
    logger.error("params yaml file not found")
    raise
except yaml.YAMLError as exc:
    logger.error(f"error in params.yaml file format: {exc}")
    raise
except Exception as e:
    logger.error(e)
    raise
final_data_dir = os.path.join("data", "final")

try:
    train_dataset = pd.read_csv (os.path.join(final_data_dir, "train.csv"))
    logger.info("train dataset read successfully")

except FileNotFoundError as e:
    logger.error("train dataset file not found")
    raise
except Exception as e:
    logger.error(e)
    raise
try:
    test_dataset = pd.read_csv (os.path.join(final_data_dir, "test.csv"))
    logger.info("test dataset read successfully")

except FileNotFoundError as e:
    logger.error("test dataset file not found")
    raise
except Exception as e:
    logger.error(e)
    raise
try:
    with open(os.path.join("models","transformer.pkl"), "rb") as f:
        transformer=pickle.load(f)

except FileNotFoundError as e:
    logger.error("transformer pkl not found")
    raise
except pickle.PickleError as e:
    logger.error(f"error in pkl file: {e}")
    raise
except Exception as e:
    logger.error(e)
    raise
X_train,y_train,X_test,y_test = train_dataset.iloc[:,:-1], train_dataset.iloc[:,-1], test_dataset.iloc[:,:-1], test_dataset.iloc[:,-1]

try:
    model = LGBMClassifier(
        **params["model_parameters"]
    )
except Exception as e:
    logger.error(e)
    raise

model.fit (X_train, y_train)
logger.info("model fit successfully")

y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

try:
    with open(os.path.join("reports","evaluate_matrices.json"),"w") as f:
        report={
            "train": classification_report(y_train,y_train_hat,output_dict=True),
            "test": classification_report(y_test,y_test_hat,output_dict=True)
        }
        json.dump(report,f)
    logger.info("evaluation report saved successfully")

except FileNotFoundError as e:
    logger.error(e)
    raise
except Exception as e:
    logger.error(e)
    raise

try:
    with open(os.path.join("models","final_model.pkl"),"wb") as f:
        pickle.dump(model,f)
    logger.info("model saved successfully")

except FileNotFoundError as e:
    logger.error(e)
    raise
except Exception as e:
    logger.error(e)




