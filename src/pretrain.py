from models import get_model, get_logits
from loss import get_loss_fn
from metrics import cal_llloss_with_logits, cal_auc, cal_prauc
from utils import get_optimizer

import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from data import get_dataset

class Trainer:
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.device = 'cuda' if torch.cuda.is_available() and params["cuda"] is True else 'cpu'

        self.method = params['method']
        self.epoch = params["epoch"]
        self.batch_size = params["batch_size"]
        
        self.dataset = get_dataset(params)
        self.train_dataset = self.dataset["train"]
        self.test_dataset = self.dataset["test"]
        self.train_dataloader = self.train_dataset.get_dataloader(batch_size=params["batch_size"], \
            shuffle=True, num_workers=params["num_workers"])
        self.test_dataloader = self.test_dataset.get_dataloader(batch_size=params["batch_size"], \
            shuffle=False, num_workers=params["num_workers"])
        
        self.loss_fn = get_loss_fn(params["loss"])        
        self.optimizer = get_optimizer(params["optimizer"], params)(self.model.parameters())
        
    def save_model(self):
        model_name = self.params["data_type"] + "_model.pt"
        if self.method == "Pretrain":
            model_path = os.path.join(self.params["model_ckpt_path"], model_name)
        elif self.method == "FSIW":
            if self.params["fsiw_pretraining_type"] == "fsiw0":
                model_path = os.path.join(self.params["pretrain_fsiw0_model_ckpt_path"], model_name)
            else:
                model_path = os.path.join(self.params["pretrain_fsiw1_model_ckpt_path"], model_name)
        elif self.method == "DDFM":
            model_path = os.path.join(self.params["pretrain_ddfm_model_ckpt_path"], model_name)
        else:
            raise NotImplementedError()
        torch.save(self.model.state_dict(), model_path)
        return
        
    def train_epoch(self, epoch):
        self.model.train()
        self.model.to(self.device)
        iterator_bar = tqdm(self.train_dataloader)
        loss_sum = 0.0
        step_num = len(iterator_bar)

        for batch in iterator_bar:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            logits = get_logits(self.model(batch["x"]), self.params["model"])
            loss = self.loss_fn({"label": batch["y"]}, logits, self.params)
            loss = loss["loss"]
            
            bar_description = "EPOCH[{}] LOSS[{:.5f}] ".format(epoch, loss.item())
            iterator_bar.set_description(bar_description)
            
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = loss_sum / step_num
        return avg_loss
    
    def train(self):
        print("Start Training")
        for epoch in range(1, self.epoch+1):
            avg_loss = self.train_epoch(epoch)
            print("--- EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))
        self.save_model()
        self.optimizer.zero_grad()
        print("Finish Training")

def run(params):
    model = get_model(data_type=params["data_type"], model=params["model"])
    trainer = Trainer(model, params)
    trainer.train()
    
if __name__ == "__main__":
    pass