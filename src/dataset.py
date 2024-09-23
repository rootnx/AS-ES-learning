from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class MWPDataset(Dataset):
    """
    initilize the dataset with the following arguments:
    data_dir: the directory of the data
    seed: the random seed used to split the data
    stage: train, val or test
    gen: the data to be generated (extractive/abstractive sentence)
    tokenizer: the tokenizer used to tokenize the data
    """
    def _split_data(self,src_file_path, dest_dir,train_ratio=0.8, val_ratio=0.1,random_state=42):
        #split the data into train, val and test
        # Load the CSV data into DataFrames
        df=pd.read_csv(src_file_path)
        if not os.path.exists(os.path.join(dest_dir,'train_qid.csv')):
            unique_ids=df['qid'].unique()
            rest_ids,train_ids=train_test_split(unique_ids,test_size=train_ratio,random_state=random_state)
            test_ids,val_ids=train_test_split(rest_ids,test_size=val_ratio/(1-train_ratio),random_state=random_state)
            train_df=df[df['qid'].isin(train_ids)]
            val_df=df[df['qid'].isin(val_ids)]
            test_df=df[df['qid'].isin(test_ids)]
            os.makedirs(dest_dir, exist_ok=True)
            train_df.to_csv(os.path.join(dest_dir,'train_{}.csv'.format(self.task_name)),index=False)
            val_df.to_csv(os.path.join(dest_dir,'val_{}.csv'.format(self.task_name)),index=False)
            test_df.to_csv(os.path.join(dest_dir,'test_{}.csv'.format(self.task_name)),index=False)
            train_df['qid'].to_csv(os.path.join(dest_dir,'train_qid.csv'),index=False,header=True)
            val_df['qid'].to_csv(os.path.join(dest_dir,'val_qid.csv'),index=False,header=True)
            test_df['qid'].to_csv(os.path.join(dest_dir,'test_qid.csv'),index=False,header=True)

        else:
            train_df=df[df['qid'].isin(pd.read_csv(os.path.join(dest_dir,'train_qid.csv'))['qid'].tolist())]
            val_df=df[df['qid'].isin(pd.read_csv(os.path.join(dest_dir,'val_qid.csv'))['qid'].tolist())]
            test_df=df[df['qid'].isin(pd.read_csv(os.path.join(dest_dir,'test_qid.csv'))['qid'].tolist())]
            train_df.to_csv(os.path.join(dest_dir,'train_{}.csv'.format(self.task_name)),index=False)
            val_df.to_csv(os.path.join(dest_dir,'val_{}.csv'.format(self.task_name)),index=False)
            test_df.to_csv(os.path.join(dest_dir,'test_{}.csv'.format(self.task_name)),index=False)




    def __init__(self, **kwargs):
        self.data_dir = kwargs["data_dir"]
        self.seed=kwargs["seed"]
        self.stage=kwargs["stage"]
        self.tokenizer=kwargs["tokenizer"]
        self.gen=kwargs["gen_target"]
        self.max_length=kwargs["max_length"]
        self.src_file_path=kwargs["src_file_path"]
        self.task_name=kwargs["task_name"]
        #check if the data has been split
        dest_dir=os.path.join(self.data_dir,str(self.seed))
        # if not os.path.exists(os.path.join(dest_dir,'train_{}.csv'.format(self.task_name))):
        self._split_data(src_file_path=self.src_file_path,dest_dir=dest_dir)
        data=pd.read_csv(os.path.join(dest_dir,"{}_{}.csv".format(self.stage,self.task_name)))
        self.data={}
        #tokenize
        key_mapping={
            "es": {
                "input_ids": "question",
                "labels": "answer_piece"
            },
            "as":{
                "input_ids": "question",
                "labels": "answer_piece"
            },
            "cs":{
                "input_ids": "question",
                "labels": "answer"
            },
            "cs_piece":{
                "input_ids": "question",
                "labels": "answer_piece"
            },
            "all":{
                "input_ids": "question",
                "labels": "answer_piece"
            }
        }
        inputs=data[key_mapping[self.gen]["input_ids"]].tolist()
        labels=data[key_mapping[self.gen]["labels"]].tolist()
        #tokenize
        self.data["input_ids"]=[]
        self.data["labels"]=[]
        self.data["attention_mask"]=[]
        for i in range(len(inputs)):
            input=self.tokenizer(inputs[i],return_tensors="pt",max_length=self.max_length,truncation=True)
            input_ids=input["input_ids"].squeeze()
            attention_mask=input["attention_mask"].squeeze()
            label_ids=self.tokenizer(labels[i],return_tensors="pt",max_length=self.max_length,truncation=True)["input_ids"].squeeze()
            label_ids[label_ids==0]=-100
            self.data["input_ids"].append(input_ids)
            self.data["labels"].append(label_ids)
            self.data["attention_mask"].append(attention_mask)



    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, idx):

        return {
            "input_ids": self.data["input_ids"][idx],
            "labels": self.data["labels"][idx],
            "attention_mask": self.data["attention_mask"][idx]
        }
