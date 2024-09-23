import torch
from src.utils import *
from src.config import *
import pandas as pd
from tqdm import tqdm
import re

def get_impression_piece(**kwargs):

    """
    generate answer_piece from anomly (separated by comma)
    answer_piece.csv: id, question, answer_piece (id is the corresponding id of the question)
    """
    data=pd.read_csv(kwargs['cs_path'])
    anomaly=data['question'].tolist()
    impression=data['answer'].tolist()
    id=data['qid'].tolist()
    impression_pieces=[]
    anomaly_=[]
    id_=[]
    question=[]
    #calculate the entropy and show the progress
    for i in tqdm(range(len(anomaly))):
        # Regular expression pattern to identify sentences without splitting floats
        pattern = r'(?<!\d)\.|\.(?!\d|\.)'
        impression_piece = re.split(pattern, impression[i])
        impression_piece = [item for sublist in impression_piece for item in sublist.split('\n')]
        impression_piece = [s.strip() for s in impression_piece if s.strip()]
        impression_piece = [item for sublist in impression_piece for item in sublist.split(',')]
        impression_piece = [s.strip() for s in impression_piece if s.strip()]
        impression_piece = [ip + ',' for ip in impression_piece]
        anomaly_tmp=anomaly[i]
        for j in range(len(impression_piece)):
            if len(impression_piece[j].split())>3:
                impression_pieces.append(impression_piece[j])
                if j>0:
                    anomaly_tmp = anomaly_tmp+' '+impression_piece[j-1]
                anomaly_.append(anomaly_tmp)
                question.append(anomaly[i])
                id_.append(id[i])
    pd.DataFrame({'qid':id_,'question':question,
                  'answer_piece':impression_pieces,}).to_csv(kwargs['piece_path'],index=False)


def calculate_sentence_entropy(tokenizer,model, inputs,targets,max_length=512):
    # Tokenize the sentences
    encoding_input = tokenizer(inputs, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    encoding_target = tokenizer(targets, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    
    input_ids = encoding_input['input_ids'].cuda()
    target_ids = encoding_target['input_ids'].cuda()
    target_ids[target_ids == 0] = -100
    
    attention_mask = encoding_input['attention_mask'].cuda()
    log_mask=encoding_target['attention_mask'].cuda()
    # Generate output from the model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=target_ids, attention_mask=attention_mask, return_dict=True)

    # Calculate the probabilities for each token in the output sequence
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Mask the probabilities to keep only actual tokens, not padding
    probs_masked = probs * log_mask.unsqueeze(-1)
    log_mask = (probs_masked > 0).type(probs_masked.dtype)
    entropy = -torch.sum(probs_masked * torch.log(probs_masked) * log_mask, dim=-1)
    entropy[entropy != entropy] = 0
    entropy_summed = entropy.sum(dim=1).cpu().numpy()
    entropy_avg=[]
    for i in range(len(targets)):
        entropy_avg.append(entropy_summed[i]/len(targets[i].split()))
    return entropy_summed, entropy_avg




def entropy_cal(**kwargs):
    # Load the model
    tokenizer=TOKENIZER_TYPE[kwargs['model_type']].from_pretrained(TOKENIZER_NAME[kwargs['model_type']])
    model = MODEL_TYPE[kwargs.get('model_type')].from_pretrained(kwargs.get('model_name'))
    model.eval()
    model=model.cuda()
    # Draw the entropy distribution
    data=kwargs['data']
    impression=data['answer_piece'].tolist()
    anomaly=data['question'].tolist()
    id=data['qid'].tolist()
    batch_size = 24
    entropies_sum=[]
    entropies_avg=[]
    num_batches = len(impression) // batch_size
    for batch_idx in tqdm(range(num_batches)):
        start = batch_idx * batch_size
        end = start + batch_size
        impression_batch = impression[start:end]
        anomaly_batch=anomaly[start:end]
        h0, h1 = calculate_sentence_entropy(tokenizer,model, anomaly_batch,impression_batch)
        entropies_sum.extend(h0)
        entropies_avg.extend(h1)
    length=len(entropies_sum)
    # save data
    pd.DataFrame({'qid':id[:length],
        'question':anomaly[:length],'answer_piece':impression[:length], 'entropies_sum':entropies_sum,'entropies_avg':entropies_avg}).to_csv(kwargs['entropy_path'],index=False)
    print('done')
