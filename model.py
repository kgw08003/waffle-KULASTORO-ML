# pip install torchtext==0.6
import os
import torch
import random
import pandas as pd
import numpy as np
import requests
import urllib.request

# 데이터 불러오기
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt", filename="ratings_test.txt")
train_file = "ratings_train.txt"
test_file = "ratings_test.txt"
columns = ['id','text','label']

train_data = pd.read_csv(train_file, sep='\t', names=columns, skiprows=1).dropna() # null데이터 삭제
test_data = pd.read_csv(test_file, sep='\t', names=columns, skiprows=1).dropna()

# 랜덤 시드 고정
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# bert-base-multilingual-cased 토크나이저를 사용
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 토큰화
tokens = tokenizer.tokenize('내일은 드디어 주말이 시작되는 날입니다.')
indexes = tokenizer.convert_tokens_to_ids(tokens)

# 토큰 학습
init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)
max_input_length = tokenizer.max_model_input_sizes['bert-base-multilingual-cased']

# 토크나이저의 문장 시작 토큰과 마지막 토큰을 제거
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

# 필드 정의
from torchtext import data
TEXT = data.Field(batch_first = True,
                 use_vocab = False,
                 tokenize = tokenize_and_cut,
                 preprocessing = tokenizer.convert_tokens_to_ids,
                 init_token = init_token_idx,
                 eos_token = eos_token_idx,
                 pad_token = pad_token_idx,
                 unk_token = unk_token_idx)
LABEL = data.LabelField(dtype = torch.float)

# 데이터 분리
fields = {'text': ('text',TEXT), 'label': ('label',LABEL)}

current_dir = os.path.dirname(os.path.abspath(__file__))

train_data_path = os.path.join(current_dir, 'train_data.csv')
test_data_path = os.path.join(current_dir, 'test_data.csv')

train_data, test_data = data.TabularDataset.splits(
                            path = current_dir,
                            train = 'train_data.csv',
                            test = 'test_data.csv',
                            format = 'csv',
                            fields = fields,
)
train_data, valid_data = train_data.split(random_state=random.seed(SEED))

LABEL.build_vocab(train_data)

# 이터레이터 생성
BATCH_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    device = device)

# 모델 생성
from transformers import BertModel
bert = BertModel.from_pretrained('bert-base-multilingual-cased')

# 모델 정의
import torch.nn as nn
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
        #text = [batch_size, sent_len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        #embedded = [batch_size, sent_len, emb_dim]

        _, hidden = self.rnn(embedded)
        #hideen = [n_layers * n_directions, batch_size, emb_dim]

        if self.rnn.bidirectional:
            # 마지막 레이어의 양방향 히든 벡터만 가져옴
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        #hidden = [batch_size, hid_dim]

        output = self.out(hidden)
        #output = [batch_size, out_dim]

        return output

# 하이퍼파라미터 지정    
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, BIDIRECTIONAL, DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    

# bert 모델 훈련 x
for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False
        
# 모델 훈련
import torch.optim as optim
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)
    return acc

# train 함수 정의
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1) # output_dim = 1
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# 시간 확인 함수
import time
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
  
# 토크나이저 설정에서 attention_mask 생성
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                          attention_mask=True)

# transformer 모델 훈련
N_EPOCHS = 1
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

model.load_state_dict(torch.load('tut6-model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

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

text = input("감정 분석을 수행할 텍스트를 입력하세요: ")

# 감정 예측 및 출력
print("입력한 텍스트의 감정은:", predict_sentiment(model, tokenizer, text))

import torch
import torch.nn as nn
from transformers import BertModel

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()

        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
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