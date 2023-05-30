from models import get_model, get_logits
from loss import get_loss_fn
from utils import get_optimizer, ScalarMovingAverage
from metrics import cal_llloss_with_logits, cal_llloss_with_prob, cal_auc, cal_prauc, cal_ks
from data import get_dataset_stream
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import time
import os

class Trainer:
    def __init__(self, models, params):
        self.model = models["model"]
        self.models = models
        self.params = params
        self.method = params["method"]
        self.device = 'cuda' if torch.cuda.is_available() and params["cuda"] is True else 'cpu'

        self.epoch = params["epoch"]
        self.batch_size = params["batch_size"]
        
        self.dataset = get_dataset_stream(params)
        self.train_dataset = self.dataset["train"] # list
        self.test_dataset = self.dataset["test"] # list
        
        self.loss_fn = get_loss_fn(params["loss"])
        
        self.auc_ma = ScalarMovingAverage()
        self.nll_ma = ScalarMovingAverage()
        self.prauc_ma = ScalarMovingAverage()
        self.ks_ma = ScalarMovingAverage()

    def get_logits_dict(self, batch_x):
        logits = self.model(batch_x)
        if self.method == "FSIW":
            logits0 = self.models["fsiw0"](batch_x)
            logits1 = self.models["fsiw1"](batch_x)
            logits_dict = {"logits": logits, "logits0": logits0, "logits1": logits1}
        elif self.method == "DDFM":
            logitsx = self.models["ddfm"](batch_x)
            logitsx = get_logits(logitsx, "MLP_tn_dp")
            logits_dict = {"logits": logits, \
                "tn_logits":logitsx["tn_logits"], "dp_logits":logitsx["dp_logits"]}
        else:
            logits_dict = {"logits": logits}       
        return logits_dict
        
    def train_epoch(self, timestep, train_dataset):
        self.model.train()
        self.model.to(self.device)
        loss_sum = 0.0
        train_dataloader = train_dataset.get_dataloader(batch_size=self.params["batch_size"], \
            shuffle=True, num_workers=self.params["num_workers"])
        step_num = len(train_dataloader)
        
        for batch in train_dataloader:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            logits_dict = self.get_logits_dict(batch["x"])

            loss = self.loss_fn({"label": batch["y"], "delay_label" : batch["delay_label"]}, \
                logits_dict, self.params)
            loss = loss["loss"]
            
            bar_description = "TIMESTEP[{}] LOSS[{:.5f}] ".format(timestep, loss.item())
            loss_sum += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        avg_loss = loss_sum / step_num
        return avg_loss

    @torch.no_grad()
    def eval_test(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        all_logits = []
        all_probs = []
        all_labels = []
        for batch in dataloader:
            for k in batch.keys():
                batch[k] = batch[k].to(self.device)
            logits = self.model(batch["x"])
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["y"].cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_logits = np.reshape(np.concatenate(all_logits, axis=0), (-1, 1))
        all_labels = np.reshape(np.concatenate(all_labels, axis=0), (-1, 1))
        all_probs = np.reshape(np.concatenate(all_probs, axis=0), (-1, 1))
        if self.method == "FNC":
            all_probs = all_probs / (1-all_probs+1e-8)
            llloss = cal_llloss_with_prob(all_labels, all_probs)
        else:
            llloss = cal_llloss_with_logits(all_labels, all_logits)
        auc = cal_auc(all_labels, all_probs)
        prauc = cal_prauc(all_labels, all_probs)
        ks = cal_ks(all_labels, all_probs)
        return llloss, auc, prauc, ks, len(all_labels)
    
    def train(self):
        for _, model in self.models.items():
            model.eval()
            model.to(self.device)
        print("Start Training")
        self.timesteps = len(self.train_dataset)
        for i, (train_dataset, test_dataset) in enumerate(zip(self.train_dataset, self.test_dataset), 0):
            # finetune
            self.optimizer = get_optimizer(self.params["optimizer"], self.params)(self.model.parameters())
            s_time = time.time()
            for epoch in range(1, self.epoch+1):
                if self.params["method"] != "Pretrain":
                    avg_loss = self.train_epoch(i, train_dataset)
                    self.last_epoch_avg_loss = avg_loss
                    print("--- EPOCH[{}] AVG_LOSS[{:.5f}]".format(epoch, avg_loss))
            print("Finetune time is {} s".format(time.time()-s_time))
            self.optimizer.zero_grad()
                
            test_dataloader = test_dataset.get_dataloader(batch_size=self.params["batch_size"], \
                shuffle=False, num_workers=self.params["num_workers"])
            llloss, auc, prauc, ks, all_len = self.eval_test(test_dataloader)
            print("TIMESTEP[{}]: AUC: {:.5f} \t PRAUC: {:.5f} \t LLLOSS: {:.5f} \t KS: {:.5f}".format(i, auc, prauc, llloss, ks))
            self.auc_ma.add(auc, all_len)
            self.nll_ma.add(llloss, all_len)
            self.prauc_ma.add(prauc, all_len)
            self.ks_ma.add(ks, all_len)
         
        print("AUC_LEN: {:.5f}, \t PRAUC_LEN: {:.5f}, \t LLLOSS_LEN {:.5f}, \t KS_LEN: {:.5f}".format( \
                self.auc_ma.get_len_weight(), self.prauc_ma.get_len_weight(), self.nll_ma.get_len_weight(), self.ks_ma.get_len_weight()))
        print("Finish Training")

def stream_run(params):
    model_name = params["data_type"] + "_model.pt"
    model = get_model(data_type=params["data_type"])
    model_path = os.path.join(params["model_ckpt_path"], model_name)
    model.load_state_dict(torch.load(model_path))
    models = {"model":model}
    if params["method"] == "FSIW":
        fsiw0_model = get_model(data_type=params["data_type"])
        fsiw0_model.load_state_dict(torch.load(os.path.join(params["pretrain_fsiw0_model_ckpt_path"], model_name)))
        fsiw1_model = get_model(data_type=params["data_type"])
        fsiw1_model.load_state_dict(torch.load(os.path.join(params["pretrain_fsiw1_model_ckpt_path"], model_name)))
        models["fsiw0"] = fsiw0_model
        models["fsiw1"] = fsiw1_model
    elif params["method"] == "DDFM":
        ddfm_model = get_model(data_type=params["data_type"], model="MLP_tn_dp")
        ddfm_model.load_state_dict(torch.load(os.path.join(params["pretrain_ddfm_model_ckpt_path"], model_name)))
        models["ddfm"] = ddfm_model

    trainer = Trainer(models, params)
    trainer.train()
    
if __name__ == "__main__":
    pass

