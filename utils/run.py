import argparse
import os
from utils.entropy_logits_generate import *
from utils.postprocess import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='mt5')
    parser.add_argument('--model_name', type=str, default='google/mt5-small')
    parser.add_argument('--cs_path', type=str, default='data/MWP/math.csv')
    parser.add_argument('--piece_path', type=str, default='data/MWP/piece.csv')
    parser.add_argument('--entropy_path', type=str, default='data/MWP/entropy.csv')
    parser.add_argument('--data_dir', type=str, default='data/MWP')
    parser.add_argument('--ratio', type=float, default=1.4)
    parser.add_argument('--mode', type=int, default=1)
    parser.add_argument('--es_path', type=str, default='data/MWP/es.csv')
    parser.add_argument('--as_path', type=str, default='data/MWP/as.csv')
    parser.add_argument('--piece_entropy_path', type=str, default='data/MWP/piece_entropy.csv')
    parser.add_argument('--seg_name', type=str, default='base')

    return parser.parse_args()


def main(**kwargs):

    data=pd.read_csv(kwargs['piece_path'])
    entropy_cal(data=data,**kwargs)
    ratio_seg(**kwargs)
    get_ases_dataset(**kwargs)
        

if __name__ == "__main__":
    main(**vars(get_args()))
