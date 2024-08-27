import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
class transfer_dataset(Dataset):

    def __init__(self, embed_path,labels_path):
        embeds = os.listdir(embed_path)
        label_df = pd.read_csv(labels_path,index_col=0)
        self.qurry = []
        self.data = []
        self.label_list = []

        embed_list = [torch.load(os.path.join(embed_path,embed)).cpu().numpy() for embed in embeds]
        for i in range(len(embed_list)):
            for j in range(len(embed_list)):
                if i!=j:
                    self.qurry.append(embed_list[i])
                    self.data.append(embed_list[j])
                    self.label_list.append(label_df.loc[embeds[i],embeds[j]])


    def __len__(self):
        return len(self.qurry)

    def __getitem__(self, index):
        return self.qurry[index],self.data[index],self.label_list[index]
    
    def collate_fn(self, batch_data):
        qurry = torch.tensor(np.array([u[0] for u in batch_data]))
        data = torch.tensor(np.array([u[1] for u in batch_data]))
        label = torch.tensor(np.array([u[2] for u in batch_data]))

        return qurry,data,label