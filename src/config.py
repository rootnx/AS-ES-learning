from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_TYPE={
    't5':T5ForConditionalGeneration,
    'bart':BartForConditionalGeneration,
    'mt5':MT5ForConditionalGeneration,
    'ft5':AutoModelForSeq2SeqLM
}
TOKENIZER_TYPE={
    't5':T5Tokenizer,
    'bart':BartTokenizer,
    'mt5':MT5Tokenizer,
    'ft5':AutoTokenizer
}

TOKENIZER_NAME={
    't5':'t5-base',
    'bart':'facebook/bart-base',
    'mt5':'output/mt5-base',
    'ft5':'output/flan-t5-base'
}