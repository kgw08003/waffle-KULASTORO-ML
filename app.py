from flask import Flask, request, jsonify

app = Flask(__name__)
@app.route('/')
def index():
    print("hello")
    return "hello"

@app.route('/predict', methods=['GET'])
def predict_sentiment():    
    import os
    import torch
    import random
    import pandas as pd
    import numpy as np
    import requests
    import urllib.request
    import torch.nn as nn
    from transformers import BertModel
    from transformers import BertTokenizer

    bert = BertModel.from_pretrained('bert-base-multilingual-cased')

    hidden_dim = 256
    output_dim = 1
    n_layers = 2
    bidirectional = True
    dropout = 0.25

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                            attention_mask=True)

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
    max_input_length = tokenizer.model_max_length

    BATCH_SIZE = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class BERTGRUSentiment(nn.Module):
        def __init__(self, bert, hidden_dim, output_dim,
                    n_layers, bidirectional, dropout):
            super().__init__()
            self.bert = bert
            embedding_dim = bert.config.to_dict()['hidden_size']
            self.rnn = nn.GRU(embedding_dim, hidden_dim,
                            num_layers = n_layers,
                            bidirectional = bidirectional,
                            batch_first = True,
                            dropout = 0 if n_layers <2 else dropout)
            self.out = nn.Linear(hidden_dim * 2 if bidirectional
                                else hidden_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text):
            with torch.no_grad():
                embedded = self.bert(text)[0]

            _, hidden = self.rnn(embedded)

            if self.rnn.bidirectional:
                hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
            else:
                hidden = self.dropout(hidden[-1,:,:])

            output = self.out(hidden)

            return output
        
    model = BERTGRUSentiment(bert, hidden_dim, output_dim,
                            n_layers, bidirectional, dropout)

    model = model.to(device)

    model.load_state_dict(torch.load('tut6-model.pt'))

    def predict_sentiment(model, tokenizer, sentence):
        model.eval()
        tokens = tokenizer.tokenize(sentence)
        tokens = tokens[:max_input_length-2]
        indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
        tensor = torch.LongTensor(indexed).to(device)
        tensor = tensor.unsqueeze(0)

        # 모델에 입력하여 감정 예측
        with torch.no_grad():
            prediction = torch.sigmoid(model(tensor))

        # 감정 예측 범위로 분류
        sentiment_score = prediction.item()
        if sentiment_score <= 0.45:
            return "부정적"
        elif 0.45 < sentiment_score <= 0.55:
            return "중립적"
        else:
            return "긍정적"

    text = "오늘하루가 너무 힘들었다 "

    # 감정 예측 및 출력
    print("입력한 텍스트의 감정은:", predict_sentiment(model, tokenizer, text))

    text1 = predict_sentiment(model, tokenizer, text)
    
    return text1

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

# 오늘하루가 너무 힘들었다 -> 부정적
# 오늘 행복한 하루였다 -> 긍정적
