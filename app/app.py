import uvicorn
from fastapi import FastAPI
from model_update import update_model
from typing import Dict
from pydantic import BaseModel
import pickle
import numpy as np
from text_preprocess import DataPreprocessing
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse
import pandas as pd
from typing import List
from datetime import datetime
#update_model()
from wordcloud import WordCloud
from nltk.corpus import stopwords
from fastapi import FastAPI, HTTPException
import io
with open("artifacts/transformer.pkl", 'rb') as file:
    transformer = pickle.load(file)

with open("artifacts/model/model.pkl", 'rb') as file:
    model = pickle.load(file)




class Comment(BaseModel):
    text: str
    timestamp: str

class CommentData(BaseModel):
    comments: list[Comment]

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
def predict(data):
    # Extract comments from the incoming request
    comments = data.comments

    # Preprocess the comments
    preprocessed_data = [DataPreprocessing.text_preprocessing(comment) for comment in comments]

    # Transform the data for the model
    transformed_data = transformer.transform(np.array(preprocessed_data))

    # Get predictions from the model
    predicted_classes = model.predict(transformed_data).tolist()

    # Define sentiment classes
    #classes = ["negative", "neutral", "positive"]

    # Create a response with comments and their corresponding sentiments
    response = [
        {
            "comment": comments[index],
            "sentiment": sentiment_class  # Use the class names
        }
        for index, sentiment_class in enumerate(predicted_classes)
    ]

    return response  # Return the response as JSON automatically


@app.post('/predict_with_timestamps')
async def predict_with_timestamps(data: CommentData):
    comments_data = data.comments

    if not comments_data:
        raise HTTPException(status_code=400, detail="No comments data provided")

    try:
        # Extract comments and timestamps
        comments = [item.text for item in comments_data]
        timestamps = [item.timestamp for item in comments_data]

        # Preprocess the comments
        preprocessed_data = [DataPreprocessing.text_preprocessing(comment) for comment in comments]

        # Transform the data for the model
        transformed_data = transformer.transform(np.array(preprocessed_data))

        # Get predictions from the model
        predictions = model.predict(transformed_data).tolist()

        # Convert predictions to strings for consistency
        predictions = [str(pred) for pred in predictions]

        # Return the response with original comments, predicted sentiments, and timestamps
        response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
                    for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

class GCData(BaseModel):
    sentiment_counts: Dict[str, int]
@app.post('/generate_chart')
def generate_chart(data: GCData):
    try:
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return {"error": "No sentiment counts provided"}

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('2', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response using StreamingResponse
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        return {"error": f"Chart generation failed: {str(e)}"}



# Define Pydantic models (same as models.py)
class SentimentData(BaseModel):
    timestamp: datetime
    sentiment: int  # Assuming sentiment is an integer (-1 for negative, 0 for neutral, 1 for positive)

class TrendGraphRequest(BaseModel):
    sentiment_data: List[SentimentData]  # A list of SentimentData objects

@app.post('/generate_trend_graph')
async def generate_trend_graph(data: TrendGraphRequest):
    try:
        sentiment_data = data.sentiment_data

        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame([item.dict() for item in sentiment_data])
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamps are datetime objects

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('ME')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [0, 2, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[0, 1, 2]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            0: 'red',     # Negative sentiment
            1: 'gray',     # Neutral sentiment
            2: 'green'     # Positive sentiment
        }

        for sentiment_value in [0, 1, 2]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response using StreamingResponse
        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        # Log and raise an HTTP exception in case of errors
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")



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



