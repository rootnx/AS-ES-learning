from src.dataset import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse


class TransformerScheduler:
    def __init__(self, optimizer, warmup_steps, peak_lr, total_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.total_steps = total_steps
        self.current_step = 0
        self.current_lr = 0

    def step(self):
        self.current_step += 1
        self.current_lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def get_lr(self):
        step = self.current_step
        if step <= self.warmup_steps:
            return self.peak_lr * step / self.warmup_steps
        else:
            return self.peak_lr / ((step - self.warmup_steps) ** 0.5)
        
def my_collate_fn(batch,tokenizer):
    padding_value=tokenizer.pad_token_id
    input_ids=pad_sequence([item['input_ids'] for item in batch],batch_first=True,padding_value=padding_value)
    attention_mask=pad_sequence([item['attention_mask'] for item in batch],batch_first=True,padding_value=padding_value)
    labels=pad_sequence([item['labels'] for item in batch],batch_first=True,padding_value=-100)
    return {
        'input_ids':input_ids,
        'attention_mask':attention_mask,
        'labels':labels
    }


def get_dataloaders(src_file_path,data_dir,tokenizer,gen_target,batch_size,seed=42,max_length=128,**kwargs):
    collate_fn = lambda batch: my_collate_fn(batch, tokenizer)
    train_dataloader=DataLoader(MWPDataset(src_file_path=src_file_path,data_dir=data_dir,tokenizer=tokenizer,
            gen_target=gen_target,stage='train',seed=seed
            ,max_length=max_length,**kwargs), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_data_loader=DataLoader(MWPDataset(src_file_path=src_file_path,data_dir=data_dir,tokenizer=tokenizer,
            gen_target=gen_target,stage='val',seed=seed
            ,max_length=max_length,**kwargs), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    test_data_loader=DataLoader(MWPDataset(src_file_path=src_file_path,data_dir=data_dir,tokenizer=tokenizer,
            gen_target=gen_target,stage='test',seed=seed
            ,max_length=max_length,**kwargs), batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    return train_dataloader,val_data_loader,test_data_loader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mt5')
    parser.add_argument('--model_name', type=str, default='google/mt5-small')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--warmup_rate', type=float, default=0.1)
    parser.add_argument('--epoch_num', type=int, default=10)
    parser.add_argument('--data_dir', type=str, default='data/1.1')
    parser.add_argument('--src_file_path', type=str, default='data/src.csv')
    parser.add_argument('--max_length', type=int, default=300)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--save_gen', action='store_true')
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--dataset',type=str,default='PET',choices=['PET','MWP'])
    parser.add_argument('--gen_target',type=str,default='as',choices=['as','es','cs','cs_piece'],help='as: abstractive sentence, es: extractive sentence, cs: complete sentence')
    parser.add_argument('--task_name',type=str,default='test',help='task name')
    return parser.parse_args()


