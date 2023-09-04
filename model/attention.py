import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import gc
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BertForSequenceClassification
from transformers import  get_linear_schedule_with_warmup
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, matthews_corrcoef, roc_auc_score, f1_score
from scipy.special import softmax
from Bio import SeqIO
from preprocess.utils import Sequence, DNASequence, KmerSequence, compute_all_metrics



class BertNup(LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.dnabert = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels = 2,
            output_attentions = False, 
            output_hidden_states = False,)

    def forward(self,input_ids, attention_mask, labels = None):
        x = self.dnabert(input_ids, attention_mask = attention_mask)['logits']
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


    def validation_epoch_end(self, outputs) -> None:
        loss = sum(output['loss'] for output in outputs) / len(outputs)
        probas = torch.hstack([output['probas'] for output in outputs]).cpu().numpy()
        labels = torch.hstack([output['labels'] for output in outputs]).cpu().numpy()
        if 0 not in labels or 1 not in labels:
            auc = 0
        else:
            _, _, _, _, _, auc = all_metrics(probas,labels,verbose = 0)
        self.log("val_loss", loss)

        
class BertNupAttention(LightningModule):
    def __init__(self, saved_model, start_attn_layer=None, end_attn_layer=None, head=None):
        super().__init__()
        self.dnabert = BertForSequenceClassification.from_pretrained(saved_model, local_files_only = True, output_attentions = True)
        if start_attn_layer==None:
            self.start_attn_layer = 11
            self.end_attn_layer = 12
        elif end_attn_layer==None:
            self.end_attn_layer = self.start_attn_layer + 1
        else:
            self.start_attn_layer = start_attn_layer
            self.end_attn_layer = end_attn_layer
        self.head = head
        self.max_length = 147

    def forward(self,input_ids, attention_mask, labels = None):
        attention = self.dnabert(input_ids)[-1]
        attn = format_attention(attention)
        if self.head == None:
            item_attn_score = attn[self.start_attn_layer:self.end_attn_layer,:,0:1,:].sum(dim=(0,1,2))
        else:
            item_attn_score = attn[self.start_attn_layer:self.end_attn_layer,self.head:self.head+1,:,:].sum(dim=(0,1,2))
        return item_attn_score
        
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        attention = self.dnabert(input_ids)[-1]
        input_id_list = input_ids[0].tolist() # Batch index 0
        attn = format_attention(attention)
        if self.head == None:
            batch_attn_score = attn[:,self.start_attn_layer:self.end_attn_layer,:,0:1,1:self.max_length-1].sum(dim=(1,2,3))
        else:
            batch_attn_score = attn[:,self.start_attn_layer:self.end_attn_layer,self.head:self.head+1,:,1:self.max_length-1].sum(dim=(1,2,3))

        return batch_attn_score

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # batch_size x num_heads x seq_len x seq_len
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x batch_size x num_heads x seq_len x seq_len
    return torch.stack(squeezed).transpose(dim0 = 0, dim1 = 1)

def process_attention_score(attention, kmer):
    attention_scores = np.array(attention).reshape(np.array(attention).shape[0],1)
    real_scores = get_real_score(attention_scores, kmer, 'mean')
    real_scores = real_scores / np.linalg.norm(real_scores, ord=1)
    scores = real_scores.reshape(1, real_scores.shape[0])
    return scores

def get_real_score(attention_scores, kmer, metric):
    counts = np.zeros([len(attention_scores)+kmer-1])
    real_scores = np.zeros([len(attention_scores)+kmer-1])
    if metric == "mean":
        for i, score in enumerate(attention_scores):
            for j in range(kmer):
                counts[i+j] += 1.0
                real_scores[i+j] += score
        real_scores = real_scores/counts
    else:
        pass
    return real_scores

def visualize_token2token_scores(scores_mat, x_label_name='Head', head = None, xticks = range(149)):
    if head == None:
        fig = plt.figure(figsize=(20, 20))

        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx+1)
            # append the attention weights
            im = ax.imshow(scores_np, cmap='viridis')
            fontdict = {'fontsize': 10}
            ax.set_xticks(range(149))
            ax.set_yticks(range(149))
            ax.set_xlabel('{} {}'.format(x_label_name, idx))
            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize = (20,20))
        score_np = np.array(scores_mat[head])
        im = ax.imshow(score_np, cmap = 'viridis')
        ax.set_xticks(range(149))
        ax.set_yticks(range(149))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(xticks)
        ax.set_xlabel('Head {}'.format(head))
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


