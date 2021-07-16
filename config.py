import torch

load = False

vocab_size = 32000
max_len = 512
n_position = 512
batch_size = 32
model_dim = 768
ff_dim = 2048
head = 8
n_layers = 6
dropout_rate = 0.1
n_epochs = 3

seed = 116

learning_rate = 1e-5
betas = (0.9, 0.98)
max_grad_norm = 1.0

warmup = 128000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model_name = 'bert-base-uncased'

train_data = 'data/train_data.pkl'
text_data = 'data/processed_cornell_data.txt'
data_dir = 'data'
fn = 'trained_model'

# This is for training from a previous saved model in Kaggle
ckpt_path = '../../input/trained-model/trained_model.pth'
