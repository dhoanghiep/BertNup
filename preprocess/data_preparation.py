import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.model_selection import KFold
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def get_data(path_name):
    data = []
    f = SeqIO.parse(open(path_name),'fasta')
    for fasta in f:
        data.append([str(fasta.seq) , fasta.id[0]])
    return data
    
def generate_dataframe(data):
    sequence = [item[0] for item in data]
    lab = [1 if item[1] == 'n' else 0 for item in data]
    return pd.DataFrame({'sequence':sequence,'label':lab}).drop_duplicates(ignore_index = True)

def k_fold_split(data_path, random_state = 1, save_dir = None, n_splits = 10):
    data_name = data_path.split("/")[-1][:-4]
    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    raw_dataframe = generate_dataframe(get_data(data_path))
    folds_indexes = [test_index for _,test_index in kf.split(raw_dataframe, raw_dataframe.label)]
    split_dataframes = []
    k_fold_number = 0
    for i in range(n_splits):
        test_index = folds_indexes[i]
        val_index = folds_indexes[i-1]
        train_index = [i for i in range(raw_dataframe.shape[0]) if i not in np.hstack((test_index,val_index))]
        
        train_set = raw_dataframe.loc[train_index].reset_index()
        val_set = raw_dataframe.loc[val_index].reset_index()
        test_set = raw_dataframe.loc[test_index].reset_index()
        if save_dir is not None:
            save_subdir = save_dir + '/' + data_name +'/split_' + str(k_fold_number) + '/'
            os.makedirs(save_subdir, exist_ok = True)
            train_set.to_csv(save_subdir + 'train.csv')
            val_set.to_csv(save_subdir + 'val.csv')
            test_set.to_csv(save_subdir + 'test.csv')
        else:
            split_dataframes.append((train_set, val_set, test_set))
        k_fold_number += 1   
    if save_dir is not None:        
        print(f'Split {data_path} to {n_splits} groups, saved to {save_dir + "/" + data_name}')
    else:
        return split_dataframes