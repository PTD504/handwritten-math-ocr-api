import re
import json
from config import config

def tokenize_latex(formula: str) -> list:
    token_pattern = r'(\\[a-zA-Z]+|[{}_^$%&#]|[0-9]+|[a-zA-Z]+|[^\s])'
    tokens = re.findall(token_pattern, formula)
    return tokens

def load_vocab(filename: str = 'vocab.json') -> tuple:
    with open(f'{config.model_dir}/{filename}', 'r', encoding='utf-8') as f:
        data = json.load(f)
    vocab = data['vocab']
    idx2char = {int(k): v for k, v in data['idx2char'].items()}
    return vocab, idx2char

def tokens_to_latex(token_ids: list, idx2char: dict) -> str:
    filtered_ids = [tid for tid in token_ids if tid in idx2char and idx2char[tid] not in [config.sos_token, config.eos_token, config.pad_token]]
    latex = ' '.join([idx2char[tid] for tid in filtered_ids])
    return latex

def clean_latex_output(latex_str):
    latex_str = re.sub(r'\\begin\s+\{', r'\\begin{', latex_str)
    latex_str = re.sub(r'\\end\s+\{', r'\\end{', latex_str)
    latex_str = re.sub(r'\{(\s+)([a-zA-Z]+)(\s+)\}', r'{\2}', latex_str)
    return latex_str
