# pip install torchtext==0.6
import os
import torch
import random
import pandas as pd
import numpy as np

columns = ['id','text','label']
train_data = pd.read_csv('ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna()
test_data = pd.read_csv('ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

train_data.to_csv('train_data.csv',index=False)
test_data.to_csv('test_data.csv',index=False)

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

tokens = tokenizer.tokenize('내일은 드디어 주말이 시작되는 날입니다.')
indexes = tokenizer.convert_tokens_to_ids(tokens)

init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

max_input_length = 128
def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length-2]
    return tokens

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
    
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert, HIDDEN_DIM, OUTPUT_DIM,
                        N_LAYERS, BIDIRECTIONAL, DROPOUT)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)    

for name, param in model.named_parameters():

    if name.startswith('bert'):
        param.requires_grad = False