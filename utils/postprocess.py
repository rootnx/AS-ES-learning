import pandas as pd
import os
import pdb
from tqdm import tqdm

def ratio_seg(**kwargs):
    data_dir=kwargs['data_dir']
    df = pd.read_csv(kwargs['entropy_path']) 
    keys=['entropies_sum','entropies_avg']
    ratio=kwargs['ratio']
    seg_name=kwargs['seg_name']
    os.makedirs('{}/{}'.format(data_dir,ratio),exist_ok=True)
    # calculate the average score for each id
    
    for key in tqdm(keys,desc='Processing key'):
        average_scores = df.groupby('qid')[key].mean()

        # create a new dataframe to store the result
        result_df = pd.DataFrame()

        for id, avg_score in tqdm(average_scores.items(),desc='Processing id'):
            # get the samples with the same id
            id_df = df[df['qid'] == id]
            
            # calculate the threshold (average score + 30%)
            threshold = avg_score * ratio
            id_df = df[df['qid'] == id].copy()
            # get the samples that surpass the threshold
            id_df.loc[id_df[key] > threshold, 'tag'] = 'as'
            id_df.loc[id_df[key] <= threshold, 'tag'] = 'es'
            
            # append these samples to the result dataframe
            result_df = pd.concat([result_df, id_df], ignore_index=True)

        # save the result dataframe to a new CSV file

        result_df.to_csv('{}/{}/{}_{}_{}.csv'.format(data_dir,ratio,ratio,key,seg_name), index=False)


def get_ases_dataset(piece_entropy_path,es_path,as_path,all_path):
    # as/es/all file segmentation for normal
    # generate answer|AS +ES / answer|ES + AS pairs
    data=pd.read_csv(piece_entropy_path)
    # a new dataframe
    es_rows=[]
    as_rows=[]
    all_rows=[]
    # iterate over each row
    current_id=data['qid'][0]
    current_query=data['question'][0]
    for index, row in tqdm(data.iterrows(),desc='Processing row'):
        if row['qid']==current_id:
            row['question']=current_query
        else:            
            current_id=row['qid']
            current_query=row['question']
        if row['tag']=='es':
            es_rows.append(row.to_dict())
        else:
            as_rows.append(row.to_dict())
        all_rows.append(row.to_dict())
        current_query=current_query+','+row['answer_piece']
    df_es=pd.DataFrame(es_rows)
    df_as=pd.DataFrame(as_rows)
    df_all=pd.DataFrame(all_rows)
    df_es.to_csv(es_path)
    df_as.to_csv(as_path)
    df_all.to_csv(all_path)
    