from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pandas as pd
import re
from tqdm import tqdm
import pdb
import argparse
import torch
from sacrebleu import corpus_bleu

device='cuda' if torch.cuda.is_available() else 'cpu'

def two_stage_gen(input,tokenizer,model_es,model_as):
    answer_length=len(input)
    input_ids=tokenizer(input, return_tensors='pt').input_ids.to(device)
    loop_count=0
    while 'answer is' not in input and loop_count<5:
        model_es_outputs=model_es.generate(input_ids=input_ids,max_new_tokens=100)
        input=input+' '+tokenizer.decode(model_es_outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if 'answer is' in input:
            break
        input_ids=tokenizer(input, return_tensors='pt').input_ids.to(device)
        model_as_outputs=model_as.generate(input_ids=input_ids, max_new_tokens=100)
        input=input+' '+tokenizer.decode(model_as_outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_ids=tokenizer(input, return_tensors='pt').input_ids.to(device)
        loop_count+=1
    return input[answer_length:]


def main(**kwargs):
    es_path=kwargs['es_path']
    as_path=kwargs['as_path']
    tokenizer=AutoTokenizer.from_pretrained('output/flan-t5-base')
    model_es=AutoModelForSeq2SeqLM.from_pretrained(es_path)
    model_es.to(device)
    model_es.eval()
    model_as=AutoModelForSeq2SeqLM.from_pretrained(as_path)
    model_as.to(device)
    model_as.eval()
    df=pd.read_csv('data/MWP/1.4/42/test_cs.csv')
    pattern = "\d+[.,/]?\d*"
    # generate the output and save it to the csv file
    total=0
    correct=0
    for index,row in tqdm(df.iterrows()):

        outputs=two_stage_gen(row['question'],tokenizer,model_es,model_as)
        df.loc[index,'gen']=outputs
        try:
            pred=re.findall(pattern, outputs)[-1]
            pred=pred.split(',')[0]
        except:
            print(outputs)
            pred=522
        try:
            gold=re.findall(pattern, row['answer'])[-1]
        except:
            print(row['answer'])
            gold=648
        df.loc[index,'gold']=gold
        df.loc[index,'pred']=pred
        df.loc[index,'correct']=gold==pred
        if gold==pred:
            correct+=1
        total+=1
    print("saving to {}".format(kwargs['save_path']))
    df.to_csv(kwargs['save_path'],index=False)
    print("acc={}".format(correct/total))
    answer=df['answer'].tolist()
    gen=df['gen'].tolist()
    bleu=corpus_bleu(answer,[gen])
    print("bleu={}".format(bleu.score))
    print('done')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--es_path', type=str, default='output/MWP/as/output/MWP/lr_8e-5/best_bleu/sum/lr_8e-5/best_bleu')
    parser.add_argument('--as_path', type=str, default='output/MWP/es/output/MWP/lr_8e-5/best_bleu/sum/lr_8e-5/best_bleu')
    parser.add_argument('--save_path', type=str, default='data/MWP/1.4/42/test/test.csv')
    return parser.parse_args()

if __name__ == '__main__':
    main(**vars(get_args()))