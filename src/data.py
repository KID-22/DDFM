import math
import copy
import os

import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils import data

from utils import parse_float_arg
from criteo_data import get_criteo_data_df

SECONDS_A_DAY = 60*60*24
SECONDS_AN_HOUR = 60*60
SECONDS_DELAY_NORM = 1
SECONDS_FSIW_NORM = SECONDS_A_DAY*5
CRITEO_INPUT_SIZE = 17
# the input of neural network should be normalized

class DataDF(object):

    def __init__(self, features, click_ts, pay_ts, sample_ts=None, labels=None, delay_label=None):
        self.x = copy.deepcopy(features)
        self.click_ts = copy.deepcopy(click_ts)
        self.pay_ts = copy.deepcopy(pay_ts)
        if sample_ts is not None:
            self.sample_ts = copy.deepcopy(sample_ts)
        else:
            self.sample_ts = copy.deepcopy(click_ts)
        if labels is not None:
            self.labels = copy.deepcopy(labels)
        else:
            self.labels = (pay_ts > 0).astype(np.int32)
        if delay_label is not None:
            self.delay_label = delay_label
        else:
            self.delay_label = np.zeros_like(pay_ts)

    def sub_days(self, start_day, end_day):
        start_ts = start_day*SECONDS_A_DAY
        end_ts = end_day*SECONDS_A_DAY
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask],
                      self.delay_label[mask])

    def sub_hours(self, start_hour, end_hour):
        start_ts = start_hour*SECONDS_AN_HOUR
        end_ts = end_hour*SECONDS_AN_HOUR
        mask = np.logical_and(self.sample_ts >= start_ts,
                              self.sample_ts < end_ts)
        return DataDF(self.x[mask],
                      self.click_ts[mask],
                      self.pay_ts[mask],
                      self.sample_ts[mask],
                      self.labels[mask],
                      self.delay_label[mask])
    

    def add_fake_neg(self):
        pos_mask = self.pay_ts > 0
        x = np.concatenate(
            (copy.deepcopy(self.x), copy.deepcopy(self.x[pos_mask])))
        sample_ts = np.concatenate(
            [self.click_ts, self.pay_ts[pos_mask]], axis=0)
        click_ts = np.concatenate(
            [self.click_ts, self.click_ts[pos_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts, self.pay_ts[pos_mask]], axis=0)
        labels = copy.deepcopy(self.labels)
        labels[pos_mask] = 0
        labels = np.concatenate([labels, np.ones((np.sum(pos_mask),), dtype=int)], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])
        return DataDF(x[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx])

    def to_vallina(self):
        idx = list(range(self.x.shape[0]))
        idx = sorted(idx, key=lambda x: self.sample_ts[x])  # sort by sampling time
        return DataDF(self.x[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx],
                      self.delay_label[idx])

    def to_fsiw_1(self, cd, T):  # build pre-training dataset 1 of FSIW
        mask = np.logical_and(self.click_ts < T-cd, self.pay_ts > 0)
        x = copy.deepcopy(self.x[mask])
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.click_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < T - cd] = 1
        # FSIW needs elapsed time information
        np.insert(x, x.shape[1], (T-click_ts-cd)/SECONDS_FSIW_NORM, axis=1)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)

    def to_fsiw_0(self, cd, T):  # build pre-training dataset 0 of FSIW
        mask = np.logical_or(self.pay_ts >= T-cd, self.pay_ts < 0)
        mask = np.logical_and(self.click_ts < T-cd, mask)
        x = copy.deepcopy(self.x[mask])
        pay_ts = self.pay_ts[mask]
        click_ts = self.click_ts[mask]
        sample_ts = self.sample_ts[mask]
        label = np.zeros((x.shape[0],))
        label[pay_ts < 0] = 1
        # x.insert(x.shape[1], column="elapse", value=(
        #     T-click_ts-cd)/SECONDS_FSIW_NORM)
        np.insert(x, x.shape[1], (T-click_ts-cd)/SECONDS_FSIW_NORM, axis=1)
        return DataDF(x,
                      click_ts,
                      pay_ts,
                      sample_ts,
                      label)
 
    def to_fsiw_tune(self, cut_ts):
        label = np.logical_and(self.pay_ts > 0, self.pay_ts < cut_ts)
        np.insert(self.x, self.x.shape[1], (cut_ts - self.click_ts)/SECONDS_FSIW_NORM, axis=1)
        return DataDF(self.x,
                      self.click_ts,
                      self.pay_ts,
                      self.sample_ts,
                      label)

    def shuffle(self):
        idx = list(range(self.x.shape[0]))
        np.random.shuffle(idx)
        return DataDF(self.x[idx],
                      self.click_ts[idx],
                      self.pay_ts[idx],
                      self.sample_ts[idx],
                      self.labels[idx])

    def add_ddfm_duplicate_samples(self, cut_sec, rn_win):
        inw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts <= cut_sec) # pay in window
        onw_mask = np.logical_and(self.pay_ts > 0, self.pay_ts - self.click_ts > cut_sec)
        neg_dup_mask = np.logical_or(self.pay_ts < 0, self.pay_ts - self.click_ts > rn_win)
        df1 = copy.deepcopy(self.x) # observe data
        df2 = copy.deepcopy(self.x) # duplicate data
        x = np.concatenate([df1[inw_mask], df1[~inw_mask], df2[onw_mask], df2[neg_dup_mask]])
        sample_ts = np.concatenate(
            [self.click_ts[inw_mask]+cut_sec, self.click_ts[~inw_mask]+cut_sec, 
            self.pay_ts[onw_mask], self.click_ts[neg_dup_mask] + rn_win], axis=0)
        click_ts = np.concatenate([self.click_ts[inw_mask], self.click_ts[~inw_mask], \
            self.click_ts[onw_mask], self.click_ts[neg_dup_mask]], axis=0)
        pay_ts = np.concatenate([self.pay_ts[inw_mask], self.pay_ts[~inw_mask], \
            self.pay_ts[onw_mask], self.pay_ts[neg_dup_mask]], axis=0)
        labels = np.concatenate([np.ones((np.sum(inw_mask),)), np.zeros((np.sum(~inw_mask),)), \
            np.ones((np.sum(onw_mask),)), np.zeros((np.sum(neg_dup_mask),))], axis=0)
        delay_label = np.concatenate([np.zeros((np.sum(inw_mask),)), np.zeros((np.sum(~inw_mask),)), \
            np.ones((np.sum(onw_mask),)), np.ones((np.sum(neg_dup_mask),))], axis=0)
        idx = list(range(x.shape[0]))
        idx = sorted(idx, key=lambda x: sample_ts[x])  # sort by sampling time
        return DataDF(x[idx],
                      click_ts[idx],
                      pay_ts[idx],
                      sample_ts[idx],
                      labels[idx],
                      delay_label[idx])

def get_dataset_stream(params):
    name = params["dataset"]
    train_stream, test_stream = [], []
    dm, de = 30, 60
    rn_win = params["rn_win"]
    print("loading datasest {}".format(name))
    df, click_ts, pay_ts = get_criteo_data_df(params)
    if name == "last_30_train_test_oracle":
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(dm, de)
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif name == "last_30_train_test_fsiw":
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(dm, de)
        test_data = data.sub_days(dm, de)
        train_stream = []
        test_stream = []
        for tr in range(dm*24, (de-1)*24+23):
            cut_ts = (tr+1)*SECONDS_AN_HOUR
            train_hour = train_data.sub_hours(
                tr, tr+1).to_fsiw_tune(cut_ts)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif "last_30_train_test_win" in name:
        cut_hour = parse_float_arg(name, "cut_hour")
        print("cut_hour {}".format(cut_hour))
        cut_sec = cut_hour*SECONDS_AN_HOUR
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(0, de)
        mask = train_data.pay_ts - train_data.click_ts > cut_sec
        train_data.labels[mask] = 0
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, (de-1)*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif "last_30_train_test_esdfm" in name:
        cut_hour = parse_float_arg(name, "cut_hour")
        print("cut_hour {}".format(cut_hour))
        cut_sec = cut_hour*SECONDS_AN_HOUR
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(0, de).add_esdfm_cut_fake_neg(cut_sec)
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif name == "last_30_train_test_fnw":
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(dm, de).add_fake_neg()
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif name == "last_30_train_test_vanilla":
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(dm, de).to_vallina()
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            mask = train_hour.pay_ts >= (tr+1)*SECONDS_AN_HOUR
            train_hour.labels[mask] = 0
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    elif "last_30_train_test_ddfm" in name:
        cut_hour = parse_float_arg(name, "cut_hour")
        cut_sec = int(SECONDS_AN_HOUR*cut_hour)
        data = DataDF(df, click_ts, pay_ts)
        train_data = data.sub_days(0, de).add_ddfm_duplicate_samples(cut_sec, rn_win*SECONDS_A_DAY)
        test_data = data.sub_days(dm, de)
        for tr in range(dm*24, (de-1)*24+23):
            train_hour = train_data.sub_hours(tr, tr+1)
            train_stream.append(Dataset(train_hour))
        for tr in range(dm*24+1, de*24):
            test_hour = test_data.sub_hours(tr, tr+1)
            test_stream.append(Dataset(test_hour))
    else:
        raise NotImplementedError("{} data does not exist".format(name))
    return {
        "train": train_stream,
        "test": test_stream,
    }

def get_dataset(params):
    name = params["dataset"]
    data_type = params["data_type"]
    print("loading datasest {}".format(name))
    df, click_ts, pay_ts = get_criteo_data_df(params)
    data = DataDF(df, click_ts, pay_ts)
    dm, de = 30, 60
    if name == "baseline_prtrain":
        test_data = data.sub_days(dm, de)
        train_data = data.sub_days(0, dm).shuffle()
        mask = train_data.pay_ts > dm * SECONDS_A_DAY
        train_data.labels[mask] = 0
    elif "tn_dp_pretrain" in name:
        cut_hour = parse_float_arg(name, "cut_hour")
        cut_sec = int(SECONDS_AN_HOUR*cut_hour)
        train_data = data.sub_days(0, dm).shuffle()
        mask = train_data.pay_ts > dm * SECONDS_A_DAY
        train_data.pay_ts[mask] = -1
        train_label_tn = np.reshape(train_data.pay_ts < 0, (-1, 1))
        train_label_dp = np.reshape(
            train_data.pay_ts - train_data.click_ts > cut_sec, (-1, 1))
        train_label = np.reshape(train_data.pay_ts > 0, (-1, 1))
        train_data.labels = np.concatenate(
            [train_label_tn, train_label_dp, train_label], axis=1)
        test_data = data.sub_days(dm, de)
        test_label_tn = np.reshape(test_data.pay_ts < 0, (-1, 1))
        test_label_dp = np.reshape(
            test_data.pay_ts - test_data.click_ts > cut_sec, (-1, 1))
        test_label = np.reshape(test_data.pay_ts > 0, (-1, 1))
        test_data.labels = np.concatenate(
            [test_label_tn, test_label_dp, test_label], axis=1)
    elif "fsiw1" in name:
        cd = parse_float_arg(name, "cd")
        print("cd {}".format(cd))
        train_data = data.sub_days(0, dm).shuffle()
        mask = train_data.pay_ts > dm * SECONDS_A_DAY
        train_data.pay_ts[mask] = -1
        test_data = data.sub_days(dm, de)
        train_data = train_data.to_fsiw_1(
            cd=cd*SECONDS_A_DAY, T=dm*SECONDS_A_DAY)
        test_data = test_data.to_fsiw_1(
            cd=cd*SECONDS_A_DAY, T=de*SECONDS_A_DAY)
    elif "fsiw0" in name:
        cd = parse_float_arg(name, "cd")
        train_data = data.sub_days(0, dm).shuffle()
        mask = train_data.pay_ts > dm * SECONDS_A_DAY
        train_data.pay_ts[mask] = -1
        test_data = data.sub_days(dm, de)
        train_data = train_data.to_fsiw_0(
            cd=cd*SECONDS_A_DAY, T=dm*SECONDS_A_DAY)
        test_data = test_data.to_fsiw_0(
            cd=cd*SECONDS_A_DAY, T=de*SECONDS_A_DAY)
    else:
        raise NotImplementedError("{} dataset does not exist".format(name))
    return {
        "train": Dataset(train_data),
        "test": Dataset(test_data)
    }

class Dataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data.x)

    def __getitem__(self, idx):
        return {
            "x": self.data.x[idx],
            "y": self.data.labels[idx],
            "delay_label": self.data.delay_label[idx],
        }
        
    def collate_fn(self, batch):
        x = [torch.from_numpy(i["x"]).long() for i in batch]
        y = [torch.tensor([i["y"]]) for i in batch]
        delay_label = [torch.tensor([i["delay_label"]]) for i in batch]
        return {
            "x": torch.cat(x, dim=0),
            "y": torch.cat(y, dim=0),
            "delay_label": torch.cat(delay_label, dim=0),
        }
    
    def get_dataloader(self, batch_size, shuffle, num_workers):
        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, \
            num_workers=num_workers, drop_last=False, collate_fn=self.collate_fn)
