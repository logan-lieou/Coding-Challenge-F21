import torch
import torch.nn as nn

import numpy as np
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

with open("input.txt", "r") as f:
    strs = f.read()
    f.close()

tokens = tokenizer.tokenize(strs)
indexes = tokenizer.convert_tokens_to_ids(tokens)

srtsym = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
endsym = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
padsym = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
unksym = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

print(tokenizer.max_model_input_sizes['bert-base-uncased'])

def tokenize_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:512-2]
    return tokens

bert = BertModel.from_pretrained("bert-base-uncased")

class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        output = self.out(hidden)
        return output

HD = 256
OD = 1
N = 2
BI = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HD,
                         OD,
                         N,
                         BI,
                         DROPOUT)

def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:512-2]
    indexed = [srtsym] + tokenizer.convert_tokens_to_ids(tokens) + [endsym]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

print(predict_sentiment(model, tokenizer, strs))
