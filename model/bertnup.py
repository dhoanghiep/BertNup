import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
import csv
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import  get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, matthews_corrcoef, roc_auc_score, f1_score
from scipy.special import softmax
from Bio import SeqIO
from preprocess.utils import Sequence, DNASequence, KmerSequence, compute_all_metrics

class Dnabert1Dataset(Dataset):
    def __init__(self, data_path, k, max_token_length=None):
        dataframe = pd.read_csv(data_path)
        self.kmer = k
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = AutoTokenizer.from_pretrained('armheb/DNA_bert_' + str(self.kmer))
        self.sequence_length = len(self.data['sequence'][0])
        if max_token_length == None:
            self.max_token_length = self.sequence_length - k + 1 + 2
    def __getitem__(self,index):
        kmer_seq = DNASequence(self.data['sequence'].str.upper()[index]).to_kmer_sequence(self.kmer) 
        encoding = self.tokenizer(str(kmer_seq),
                                  padding='max_length',
                                  truncation = True,
                                  max_length=self.max_token_length)
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(self.data['label'][index])
        return item
    def __len__(self):
        return self.len
    
    
class BertNup(LightningModule):
    def __init__(self, pretrained_model_name=None, learning_rate=None, num_warmup_steps=None, num_training_steps=None, reinit = False):
        super().__init__()
        self.dnabert = AutoModel.from_pretrained(pretrained_model_name)
        if reinit:
            for i in range(12):
                self.dnabert.encoder.layer[-i-1].apply(self.dnabert._init_weights)        
        self.dropout1 = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768,2)
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        

    def forward(self,input_ids, attention_mask, labels = None):
        x = self.dnabert(input_ids, attention_mask = attention_mask)['pooler_output']
        x = self.dropout1(x)
        x = self.linear1(x)
        outputs = torch.softmax(x, dim = 1)[:,1]

        loss = 0
        if labels is not None:
            loss = nn.CrossEntropyLoss()(x, labels)
        return loss, outputs
        
    def predict_step(self, batch, batch_idx):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        _, probas = self(input_ids = ids,attention_mask = attention_mask, labels = labels)
        return probas
    
    def training_step(self, batch, batch_idx):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, outputs = self(input_ids = ids,attention_mask = attention_mask, labels = labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, probas = self(input_ids = ids,attention_mask = attention_mask, labels = labels)
        return {'loss':loss, 'probas':probas,'labels':labels}
    
    def configure_optimizers(self):
        optimizer  = torch.optim.AdamW(params = self.parameters(), lr = self.learning_rate, weight_decay= 0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer,self.num_warmup_steps,self.num_training_steps)
        scheduler = {"scheduler": scheduler, 
                     "interval": "step", 
                     "frequency": 1,
                    }
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs) -> None:
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        probas = torch.hstack([output['probas'] for output in outputs]).cpu().numpy()
        labels = torch.hstack([output['labels'] for output in outputs]).cpu().numpy()
        if 0 not in labels or 1 not in labels:
            auc = 0
        else:
            _, _, _, _, _, auc = compute_all_metrics(probas,labels,verbose = 0)
        self.log("val_loss", loss)
        
        
        
        
