from src.trainer import Trainer
from src.utils import *
from src.dataset import *
from src.config import *
import pdb
import wandb
import os

def main(**kwargs):
    job_id = os.environ.get("SLURM_JOB_ID")
    wandb.init(project=kwargs['dataset'],name=job_id)
    tokenizer=TOKENIZER_TYPE[kwargs['model_type']].from_pretrained(TOKENIZER_NAME[kwargs['model_type']])
    train_dataloader,val_dataloader,test_dataloader=get_dataloaders(**kwargs,tokenizer=tokenizer)
    trainer=Trainer(**kwargs)
    trainer.train(train_dataloader,val_dataloader,test_dataloader)


if __name__ == '__main__':
    main(**vars(get_args()))