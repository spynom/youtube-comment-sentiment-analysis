import uvicorn
from fastapi import FastAPI
from defined_functions import comment
import pickle
import numpy as np

with open("artifacts/transformer.pkl", 'rb') as file:
    transformer = pickle.load(file)

with open("artifacts/model/model.pkl", 'rb') as file:
    model = pickle.load(file)


app = FastAPI()
@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.post('/predict')
def predict(data:comment):
    data = data.comments
    data = transformer.transform(np.array(data))

    classes = ["negative", "neutral", "positive"]
    return {
        "prediction": [ classes[predicted_values] for  predicted_values in  model.predict(data).tolist()]
    }



if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)



