import pandas as pd
from lightgbm import LGBMClassifier
import os
import yaml
import pickle
from sklearn.metrics import classification_report
import json

with open("params.yaml") as f:
    params = yaml.safe_load(f) ["model_building"]

final_data_dir = os.path.join("data", "final")

train_dataset = pd.read_csv (os.path.join(final_data_dir, "train.csv"))
test_dataset = pd.read_csv (os.path.join(final_data_dir, "test.csv"))

with open(os.path.join("models","transformer.pkl"), "rb") as f:
    transformer=pickle.load(f)

X_train,y_train,X_test,y_test = train_dataset.iloc[:,:-1], train_dataset.iloc[:,-1], test_dataset.iloc[:,:-1], test_dataset.iloc[:,-1]

model = LGBMClassifier(
    **params["model_parameters"]
)

model.fit (X_train, y_train)

y_train_hat = model.predict(X_train)
y_test_hat = model.predict(X_test)

with open(os.path.join("reports","evaluate_matrices.json"),"w") as f:
    report={
        "train": classification_report(y_train,y_train_hat,output_dict=True),
        "test": classification_report(y_test,y_test_hat,output_dict=True)
    }
    json.dump(report,f)

with open(os.path.join("models","final_model.pkl"),"wb") as f:
    pickle.dump(model,f)




