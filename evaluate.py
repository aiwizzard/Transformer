import torch
import numpy as np
from transformers.models.bert.tokenization_bert import BertTokenizer

import config as config
from model.transformer import Transformer
from train_util import subsequent_mask


def evaluate(config, input_seq, tokenizer, model, device, verbose=True):
    r"""Evaluate the model
    
    Args:
        config: object containing the model configuration
        input_seq: input sequence
        tokenizer: Tokenizer
        model: the model to be evaluated
        verbose(bool): Determines whether to print the result
    """
    # set the model to eval model
    model.eval()
    # convert input sequence to ids
    ids = tokenizer.encode(input_seq)
    # convert token ids to tensor
    src = torch.tensor(ids, dtype=torch.long, device=device).view(1, -1)
    src_mask = torch.ones(src.size(), dtype=torch.long, device=device)
    # perform encoding and get the encoded value
    mem = model.encode(src, src_mask)
    # define the target tensor
    ys = torch.ones(1, 1).fill_(tokenizer.cls_token_id).long().to(device)
    # don't want to calculate gradients(no backprop)
    with torch.no_grad():
        for i in range(config.max_len - 1):
            # decode the encoded sequence and get the result
            out = model.decode(ys, mem, src_mask,
                             subsequent_mask(ys).type_as(ys))
            # get the probabilites
            prob = model.generate(out[:, -1])
            # get the words with the maximum probability
            _, next_word = torch.max(prob, dim = 1)
            next_word = next_word.item()
            if next_word == tokenizer.sep_token_id:
                break
            # concatenate the words with the target tensor
            ys = torch.cat([ys, torch.ones(1, 1).type_as(ys).fill_(next_word).long()], dim=1)
    ys = ys.view(-1).detach().cpu().numpy().tolist()[1:]
    # convert target tensor to text
    text = tokenizer.decode(ys)
    if verbose:
        print(f'{text}')
    return text

if __name__ == '__main__':
    # Load the model
    state_dict = torch.load(f'{config.data_dir}/{config.fn}.pth', map_location=config.device)
    # Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name)

    model = Transformer(config).to(config.device)
    model.load_state_dict(state_dict['model'])
    model.eval()
    # model.freeze()

    while True:
        s = input('You>')
        if s == 'q':
            break
        print('BOT>', end='')
        text = evaluate(config, s, tokenizer, model, config.device, True)