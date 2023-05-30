import torch
import torch.nn as nn
import torch.nn.functional as F

criteo_num_bin_size = (64, 16, 128, 64, 128, 64, 512, 512)
criteo_cate_bin_size = (512, 128, 256, 256, 64, 256, 256, 16, 256)
CRITEO_INPUT_SIZE = 17

class CriteoMLP(nn.Module):

    def __init__(self, 
                 output_size=1, 
                 hidden_sizes=[256, 256, 128], 
                 embedding_size=16,
                 dropout=0):
        super(CriteoMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.input_size = CRITEO_INPUT_SIZE
        self.embedding_num = (*criteo_cate_bin_size, *criteo_num_bin_size)
        self.embedding_size = embedding_size
        self.embeddings = nn.ModuleList()
        for i in range(len(self.embedding_num)):
            self.embeddings.append(nn.Embedding(self.embedding_num[i], self.embedding_size))
        net_input_size = len(self.embedding_num) * self.embedding_size
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(net_input_size, hidden_size))
            else:
                self.layers.append(
                    nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            self.layers.append(nn.LeakyReLU())
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        if len(self.hidden_sizes) == 0:
            self.layers.append(nn.Linear(net_input_size, output_size))
        else:
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x, _=None):
        x = x.view(-1, self.input_size)
        mlp_input = []
        for i, embedding in enumerate(self.embeddings):
            emb_i = embedding(x[:, i])
            mlp_input.append(emb_i)
        x_emb = torch.cat(mlp_input, dim=1)
        x = x_emb
        for layer in self.layers:
            x = layer(x)
        return x     

def get_model(data_type, model=None):
    if data_type == "criteo":
        if model in ["MLP_tn_dp"]:
            return CriteoMLP(output_size=2)
        else:
            return CriteoMLP()
    else:
        raise NotImplementedError()

def get_logits(logits, model):
    if model in ["MLP_tn_dp"]:
        return {
            "tn_logits": logits[:, 0].view(-1,1),
            "dp_logits": logits[:, 1].view(-1,1)
        }
    else:
        return {"logits": logits}



