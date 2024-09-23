import os
import wandb
import torch
from sacrebleu import corpus_bleu
from src.utils import *
from src.config import *
from src.dataset import *
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, AdamW
import math

class Trainer():
    """
    methods needed to be implemented:
        1.model init in __init__ (custom variables)
        2.save_model_weights
        3.cal_loss
        4.infer
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device='cpu'
        self.best_metrics_val={
            'loss':float('inf'),
            'bleu':0,
        }
        self.best_metrics_train={
            'loss':float('inf')
        }
        self.metrics_test={
        }
        self.epoch_num=0 # number of epochs to train
        self.epoch =0 # current epoch
        self.step=0
        self.lr=0
        self.current_lr=0
        self.warmup_rate=kwargs.get('warmup_rate')
        self.model_type = kwargs.get('model_type')
        self.model_name = kwargs.get('model_name')
        self.model = MODEL_TYPE[kwargs.get('model_type')].from_pretrained(kwargs.get('model_name'))
        self.tokenizer = TOKENIZER_TYPE[kwargs.get('model_type')].from_pretrained(TOKENIZER_NAME[kwargs.get('model_type')])
        self.batch_size = kwargs.get('batch_size')
        self.accumulate = kwargs.get('accumulate')
        self.epoch_num = kwargs.get('epoch_num')
        self.data_dir = kwargs.get('data_dir')
        self.seed = kwargs.get('seed')
        self.data_size=0
        self.save_path = kwargs.get('save_path')
        self.save_gen = kwargs.get('save_gen')
        self.set_attr(**kwargs)

    def save_model_weights(self,save_path):
        print('saving to model path: {}'.format(save_path))
        self.model.save_pretrained(save_path)

    def load_model_weights(self,load_path):
        self.model.from_pretrained(load_path)

    def cal_loss(self, dataloader,**kwargs):
        with torch.no_grad():
            for batch in dataloader:  
                total_loss = 0
                for k,v in batch.items():
                    batch[k]=v.to(self.device)  
                loss = self.model(**batch).loss
                total_loss += loss.item()
            loss_avg = total_loss/len(dataloader)
        return loss_avg


    def cal_bleu(self,save_path,file_name='gen.csv',**kwargs):
        data_loader=kwargs.get('dataloader')
        references = []
        hypotheses = []
        inputs=[]
        for batch in data_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["labels"].to(self.device)
            # replace -100 in the labels as we can't decode them
            targets.masked_fill_(targets == -100, self.tokenizer.pad_token_id)
            # Generate model outputs
            outputs = self.infer(input_ids)

            # Convert tensor outputs to lists of token IDs
            predicted_sequences = [self.tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]
            real_sequences = [self.tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=True) for target in targets]
            # Extend the lists
            hypotheses.extend(predicted_sequences)
            references.extend(real_sequences)
            if self.save_gen:
                input_sequences = [self.tokenizer.decode(input_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for input_id in input_ids]
                inputs.extend(input_sequences)


        # Compute BLEU score
        bleu_score = corpus_bleu(hypotheses, [references]).score
        if self.save_gen:
            print('saving to {}'.format(save_path))
            os.makedirs(save_path, exist_ok=True)
            print('saving generated results to {}'.format(os.path.join(save_path,file_name)))
            pd.DataFrame({'input':inputs,'gen':hypotheses,'target':references}).to_csv(os.path.join(save_path,file_name),index=False)

        return bleu_score

    def train(self, train_dataloader,val_dataloader, test_dataloader):
        self.model.to(self.device)
        self.data_size = len(train_dataloader)
        self.log_step = self.data_size // 10
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # self.scheduler = TransformerScheduler(self.optimizer, warmup_steps=2*self.data_size, 
                                            #   peak_lr=self.lr, total_steps=self.epoch_num*self.data_size)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=int(self.warmup_rate*self.epoch_num*self.data_size), num_training_steps=self.epoch_num*self.data_size)
        self.val_epoch = math.ceil(self.epoch_num/10)
        for epoch in tqdm(range(self.epoch_num)):
            self.model.train()
            accumulate_step = 0
            loss_avg = 0
            for batch in train_dataloader:
                # self.current_lr = self.scheduler.get_lr()
                self.current_lr = self.optimizer.param_groups[0]['lr']
                for k,v in batch.items():
                    batch[k]=v.to(self.device)                
                loss = self.model(**batch).loss
                loss = loss/self.accumulate
                loss_avg += loss.item()/len(batch)
                loss.backward()
                accumulate_step += 1
                if accumulate_step == self.accumulate:
                    accumulate_step = 0
                    # loss_avg = loss/len(batch)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    if loss_avg<self.best_metrics_train['loss']:
                        self.best_metrics_train['loss']=loss_avg
                    if self.step % self.log_step == 0:
                        print('epoch: {}, step: {}, loss: {}, lr:{}'.format(self.epoch, self.step, loss_avg,self.current_lr))
                        wandb.log({'loss': loss_avg,'lr':self.current_lr})
                    loss_avg = 0
                    self.step += 1
            self.epoch += 1
            if self.epoch % self.val_epoch == 0:
                if self.epoch<0.8*self.epoch_num:
                    self.val(metrics=['loss'],dataloader=val_dataloader,stage='val',
                             save_path=self.save_path)
                else:
                    self.val(metrics=['loss','bleu'],dataloader=val_dataloader,stage='val',
                             save_path=self.save_path)
        #test
        self.load_model_weights('{}/best_bleu'.format(self.save_path))
        self.val(metrics=['loss','bleu'],dataloader=test_dataloader,stage='test',save_path=self.save_path)

    def val(self,**kwargs):
        """
        valuate the model according to the given metrics and save the best model
        metrics: list of metrics (loss, bleu)
        dataloader: dataloader for valuation
        stage: val or test
        save_path: path to save the best model
        """
        metrics=kwargs.get('metrics')
        save_path=kwargs.get('save_path')
        stage=kwargs.get('stage')
        self.model.to(self.device)
        self.model.eval()

        if 'loss' in metrics:
            loss=self.cal_loss(**kwargs)
            if stage=='val':
                if loss<self.best_metrics_val['loss']:
                    self.best_metrics_val['loss']=loss
                    if not os.path.exists(os.path.join(save_path,'best_loss')):
                        os.makedirs(os.path.join(save_path,'best_loss'))
                    self.save_model_weights(os.path.join(save_path,'best_loss'))
                print('val_loss:{},best_val_loss:{},best_train_loss:{},step:{},epoch:{},lr:{}'.format(loss,self.best_metrics_val['loss'],self.best_metrics_train['loss'],self.step,self.epoch,self.current_lr))
                wandb.log({'val_loss':loss,'best_val_loss':self.best_metrics_val['loss'],'best_train_loss':self.best_metrics_train['loss'],'step':self.step,'epoch':self.epoch,'lr':self.current_lr})
            else:
                self.metrics_test['loss']=loss
                print('test_loss:{},step:{},epoch:{},lr:{}'.format(loss,self.step,self.epoch,self.lr))
                wandb.log({'test_loss':loss,'step':self.step,'epoch':self.epoch,'lr':self.lr})
        if 'bleu' in metrics:
            # calculate bleu
            bleu=self.cal_bleu(**kwargs)
            if stage=='val':
                if bleu>self.best_metrics_val['bleu']:
                    self.best_metrics_val['bleu']=bleu
                    if not os.path.exists(os.path.join(save_path,'best_bleu')):
                        os.mkdir(os.path.join(save_path,'best_bleu'))
                    self.save_model_weights(os.path.join(save_path,'best_bleu'))
                print('val_bleu:{},best_val_bleu:{},step:{},epoch:{},lr:{}'.format(bleu,self.best_metrics_val['bleu'],self.step,self.epoch,self.current_lr))
                wandb.log({'val_bleu':bleu,'best_val_bleu':self.best_metrics_val['bleu'],'step':self.step,'epoch':self.epoch,'lr':self.current_lr})
            else:
                self.metrics_test['bleu']=bleu
                print('test_bleu:{},step:{},epoch:{},lr:{}'.format(bleu,self.step,self.epoch,self.lr))
                wandb.log({'test_bleu':bleu,'step':self.step,'epoch':self.epoch,'lr':self.lr})
            
        

    def infer(self, input_ids):
        self.model.eval()
        input_ids.to(self.device)
        return self.model.generate(input_ids=input_ids,max_new_tokens=300)

    def set_attr(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)