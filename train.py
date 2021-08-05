import yaml
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers.models.bert.tokenization_bert import BertTokenizer

from train_util import create_masks, create_train_data, seed_everything
from dataset import ChatDataSet, SampledDataLoader
from torch.optim import AdamW
from model.transformer import Transformer
from evaluate import evaluate


def train(epoch, config, device, data_loader, toker, model, optimizer, criterion):
    # set model to train mode
    model.train()
    with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}") as pbar:
        for i, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            x, y = batch

            target = y[:, :-1]
            target_y = y[:, 1:]

            source_mask, target_mask = create_masks(x, target, toker.pad_token_id)

            out = model(x, source_mask, target, target_mask)

            optimizer.zero_grad()
            loss = criterion(out.transpose(1, 2), target_y).mean()
            loss.backward()
            optimizer.step()
            clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            pbar.update(1)
            pbar.set_postfix_str(f"loss: {loss.item():.5f}")
    # Save model for each epoch with a different name
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"{config['data_dir']}/{config['fn']}_{epoch}.pth",
    )
    # Save the final model
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"{config['data_dir']}/{config['fn']}.pth",
    )
    print("--------------------------------")
    print("Model Saved")
    print("--------------------------------")


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed_everything(config['seed'])
    toker = BertTokenizer.from_pretrained(config['bert_model_name'])
    data = create_train_data(config, toker, True)
    dataset = ChatDataSet(data)
    data_loader = SampledDataLoader(
        dataset, batch_size=config['batch_size'], padding=toker.pad_token_id
    )

    model = Transformer(config)
    model = model.to(device)

    adam_opim = AdamW(
        model.parameters(), lr=config['learning_rate'], betas=config['betas'], eps=1e-9
    )

    criterion = nn.CrossEntropyLoss(ignore_index=toker.pad_token_id, reduction="none")
    start_epoch = 0

    if config['load']:
        start_epoch = 10
        state_dict = torch.load(config['ckpt_path'], map_location=device)
        model.load_state_dict(state_dict["model"])
        adam_opim.load_state_dict(state_dict["optimizer"])

    for epoch in range(start_epoch, config['n_epochs']):
        train(epoch, config, device, data_loader, toker, model, adam_opim, criterion)
        evaluate(
            config,
            "if you accomplish your task, it is great then",
            toker,
            model,
            device,
            False,
        )
    print("Training Finished")


if __name__ == "__main__":
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    main(config)
