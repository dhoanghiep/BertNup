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
from model.bertnup import Dnabert1Dataset, BertNup

def k_fold_cv(data_name, result_dir, pretrained_model_name, kmer, learning_rate, epochs, train_params, test_params, device, all_data_dir = 'Data/Stratified_K_fold_data',start = 0,end = 10):
    data_dir = all_data_dir + '/' + data_name
    os.makedirs(result_dir, exist_ok = True)
    with open(result_dir + '/' + data_name + '.csv', "w") as f:
        f.write(",Sensitivy,Specificity,Accuracy,F1_Score,MCC,AUC\n")

    for no_split in range(start,end):
        gc.collect()
        print(f'no_split: {no_split}')
        train_path = data_dir + '/split_' + str(no_split) + '/train.csv'
        val_path = data_dir + '/split_' + str(no_split) + '/val.csv'
        test_path = data_dir + '/split_' + str(no_split) + '/test.csv'
        train_set = Dnabert1Dataset(train_path, kmer)
        val_set = Dnabert1Dataset(val_path, kmer)
        test_set = Dnabert1Dataset(test_path, kmer)
        train_loader = DataLoader(train_set, **train_params)
        val_loader = DataLoader(val_set, **test_params)
        test_loader = DataLoader(test_set, **test_params)
        
        num_training_steps = len(train_loader)*epochs
        num_warmup_steps = 40
        
        seed_everything(0)
        model = BertNup(pretrained_model_name, learning_rate, num_warmup_steps, num_training_steps)
        
        checkpoint_callback = ModelCheckpoint(dirpath = 'model_checkpoint/' + data_name, save_top_k = 1, 
                                              monitor = "val_loss", mode = "min", 
                                              save_weights_only = True)
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
        trainer = Trainer(accelerator=device, devices=1,
                          gradient_clip_val = 10,
                          max_epochs = epochs,
                          callbacks = [checkpoint_callback,early_stop_callback],
                          val_check_interval = 0.1,
                          enable_model_summary = False,                          
                         )
        trainer.fit(model = model, train_dataloaders = train_loader, val_dataloaders = val_loader)
        
        #final validation
        
        model = BertNup.load_from_checkpoint(checkpoint_path=checkpoint_callback.best_model_path,pretrained_model_name=pretrained_model_name)

        outputs = trainer.predict(model = model, dataloaders = test_loader)
        probas = torch.cat(outputs)
        probas = probas.numpy()
        labels = np.vstack(list(test_set.data.label))
        
        print('test:  ')
        sn, sp, acc, f1, mcc, auc = compute_all_metrics(probas,labels,verbose = 1)
        print(f'sn: {sn:.4f}, sp: {sp:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, mcc: {mcc:.4f}, auc: {auc:.4f}')
        results = [no_split,sn, sp, acc, f1, mcc, auc]
        with open(result_dir + '/' + data_name + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(results)

        os.remove(checkpoint_callback.best_model_path)
        
    print('######################')
    print('Results for ' + data_name + ' saved to ' + result_dir)
    print('######################')