import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Load pre-trained tokenizer and model
MODEL_NAME = "bert-base-uncased"  # Replace with your trained model if saved
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set model to evaluation mode

# Define input format
class NewsInput(BaseModel):
    text: str

# Prediction function
def predict_fake_news(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
    
    # Move tensors to GPU if available
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Run prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    return "FAKE NEWS" if prediction == 1 else "REAL NEWS"

# API route
@app.post("/predict/")
def predict_news(news: NewsInput):
    result = predict_fake_news(news.text)
    return {"prediction": result}

# Health check route
@app.get("/")
def health_check():
    return {"status": "Fake News Detector API is running!"}
