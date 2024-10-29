import uvicorn
from fastapi import FastAPI
from model_update import update_model
from pydantic import BaseModel
import pickle
import numpy as np
from text_preprocess import DataPreprocessing
from fastapi.middleware.cors import CORSMiddleware

#update_model()

with open("artifacts/transformer.pkl", 'rb') as file:
    transformer = pickle.load(file)

with open("artifacts/model/model.pkl", 'rb') as file:
    model = pickle.load(file)




class Comments(BaseModel):
    comments: list

app = FastAPI()
@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including OPTIONS
    allow_headers=["*"],  # Allows all headers
)


@app.post('/predict')
def predict(data: Comments):
    # Extract comments from the incoming request
    comments = data.comments

    # Preprocess the comments
    preprocessed_data = [DataPreprocessing.text_preprocessing(comment) for comment in comments]

    # Transform the data for the model
    transformed_data = transformer.transform(np.array(preprocessed_data))

    # Get predictions from the model
    predicted_classes = model.predict(transformed_data).tolist()

    # Define sentiment classes
    classes = ["negative", "neutral", "positive"]

    # Create a response with comments and their corresponding sentiments
    response = [
        {
            "comment": comments[index],
            "sentiment": classes[sentiment_class]  # Use the class names
        }
        for index, sentiment_class in enumerate(predicted_classes)
    ]

    return response  # Return the response as JSON automatically

@app.get('/logs')
def logs():
    logs = {
        "Server":{}
    }
    with open("server.log","r") as f:
        for line in f:
            datetime,name,level,message = line.split(" - ")
            logs["Server"][f"{datetime}"] = {"Level":level,"Message":message}

    return logs

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)



