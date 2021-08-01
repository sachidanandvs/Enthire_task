from transformers import DistilBertTokenizer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import torch
import os
import transformers
import re
device = 'cpu'
def process_tweet(tweet):
    new_tweet = tweet.lower()
    new_tweet = re.sub(r'@\w+', '', new_tweet) # Remove @s
    new_tweet = re.sub(r'#', '', new_tweet) # Remove hashtags
    #new_tweet = re.sub(r':', ' ', emoji.demojize(new_tweet)) # Turn emojis into words
    new_tweet = re.sub(r'http\S+', '',new_tweet) # Remove URLs
    new_tweet = re.sub(r'\$\S+', 'dollar', new_tweet) # Change dollar amounts to dollar
    new_tweet = re.sub(r'[^a-z0-9\s]', '', new_tweet) # Remove punctuation
    new_tweet = re.sub(r'[0-9]+', 'number', new_tweet) # Change number values to number

    inputs = tokenizer.encode_plus(
                new_tweet,
                None,
                add_special_tokens=True,
                max_length=100,
                pad_to_max_length=True,
                truncation = True,
                return_token_type_ids=True
            )
    ids = inputs['input_ids']
    mask = inputs['attention_mask']
    return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)}

model, pretrained_weights = (transformers.DistilBertModel, 'distilbert-base-uncased')
model = model.from_pretrained(pretrained_weights)

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = model
        self.drop = torch.nn.Dropout(0.1)
        self.out = torch.nn.Linear(768, 2)
    
    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output

# Creating FastAPI instance
app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello , welcome move to /docs for sentiment analysis'}

model = DistillBERTClass().to(device)

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    text : str

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
model.load_state_dict(torch.load("weights.pth"),strict=False)
model.eval()
# Creating an Endpoint to recieve the data
# to make prediction on.

class response(BaseModel):
    predicted : str

@app.post('/predict',response_model=response)
def predict(data : request_body):
    # Making the data in a form suitable for prediction
    test_data = process_tweet(data.text)
    input1 = test_data["ids"].reshape((1,100)).to(device)
    mask = test_data["mask"].reshape((1,100)).to(device)

    # Predicting the Class
    y_pred = model(input1,mask)
    _, predicted = torch.max(y_pred.data, 1)
    predicted = predicted.item()
    if predicted:
    	ans = response(predicted="positive")
    # Return the Result
    else:
    	ans = response(predicted="negative")
    return ans