import torch
import numpy as np
import pickle
from tqdm import tqdm

def create_train_data(config, tokenizer, use_pickle=False) -> list:
    r"""Create train data

    If the train data is saved as pickle object load that.
    else create, save and return the data.
    """
    data = []
    if use_pickle:
        with open(config['train_data'], 'rb') as file:
            data = pickle.load(file)
    else:
        with open(config['text_data'], 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for i in tqdm(range(0, len(lines), 3)):
            li = []
            for line in lines[i: i+2]:
                li.append(line[:config['max_len']])
            data.append(tuple(map(tokenizer.encode, li)))
        with open(config['train_data'], 'wb') as file:
            pickle.dump(data, file)
    return data


def subsequent_mask(seq):
    r""" For masking out the subsequent info. """
    _, size = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, size, size), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


def create_masks(source, target, pad=0):
    r"""Create source and target mask"""
    source_mask = (source != pad).unsqueeze(-2).to(source.device)

    target_mask = (target != pad).unsqueeze(-2).to(target.device)
    target_mask = target_mask & subsequent_mask(target)

    return source_mask, target_mask

def seed_everything(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True