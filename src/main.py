import argparse
from copy import deepcopy

import os
import random
import torch
import numpy as np

from pretrain import run
from stream_train_test import stream_run

def seed_torch(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def run_params(args):
    params = deepcopy(vars(args))
    params["model"] = "MLP_SIG"
    params["optimizer"] = "Adam"
    if args.mode == "pretrain":
        if args.method == "Pretrain":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "baseline_prtrain"
        elif args.method == "FSIW":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = args.fsiw_pretraining_type+"_cd_"+str(args.CD)
            params["model"] = "MLP_FSIW"
        elif args.method == "DDFM":
            params["loss"] = "tn_dp_pretraining_loss"
            params["dataset"] = "tn_dp_pretrain_cut_hour_"+str(args.C)
            params["model"] = "MLP_tn_dp"
        else:
            raise ValueError(
                "{} method do not need pretraining other than Pretrain".format(args.method))
    else:
        if args.method == "Pretrain":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "Oracle":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_oracle"
        elif args.method == "FSIW":
            params["loss"] = "fsiw_loss"
            params["dataset"] = "last_30_train_test_fsiw"
        elif args.method == "Vanilla_win_dup":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_esdfm_cut_hour_" + str(args.C)
        elif args.method == "Vanilla_win":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_win_cut_hour_" + str(args.C)
        elif args.method == "Vanilla":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_vanilla" 
        elif args.method == "FNW":
            params["loss"] = "fake_negative_weighted_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "FNC":
            params["loss"] = "cross_entropy_loss"
            params["dataset"] = "last_30_train_test_fnw"
        elif args.method == "DDFM":
            params["loss"] = "ddfm_loss"
            params["dataset"] = "last_30_train_test_ddfm_cut_hour_" + str(args.C)
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="delayed feedback method",
                        choices=["FSIW",
                                 "FNW",
                                 "FNC",
                                 "Pretrain",
                                 "Oracle",
                                 "Vanilla",
                                 "Vanilla_win",
                                 "Vanilla_win_dup",
                                 "DDFM"],
                        type=str, required=True)
    parser.add_argument(
        "--mode", type=str, choices=["pretrain", "stream"], help="training mode", required=True)
    parser.add_argument(
        "--data_type", type=str, choices=["criteo"], help="data type", default="criteo")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--CD", type=int, default=7,
                        help="counterfactual deadline in FSIW")
    parser.add_argument("--C", type=float, default=0.25,
                        help="elapsed time in ES-DFM")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--data_path", type=str, required=True,
                        help="path of the data.txt in criteo dataset, e.g. /home/xxx/data.txt")
    parser.add_argument("--model_ckpt_path", type=str,
                        help="path to save pretrained model")
    parser.add_argument("--pretrain_fsiw0_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw0 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_fsiw1_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained fsiw1 model,  \
                        necessary for the streaming evaluation of FSIW method")
    parser.add_argument("--pretrain_baseline_model_ckpt_path", type=str,
                        help="path to the checkpoint of pretrained baseline model(Pretrain),  \
                        necessary for the streaming evaluation of \
                            FSIW, FNW, FNC, Pretrain, Oracle, Vanilla method")
    parser.add_argument("--pretrain_ddfm_model_ckpt_path", type=str)
    parser.add_argument("--fsiw_pretraining_type", choices=["fsiw0", "fsiw1"], type=str, default="None",
                        help="FSIW needs two pretrained weighting model")
    parser.add_argument("--batch_size", type=int,
                        default=1024)
    parser.add_argument("--epoch", type=int, default=5,
                        help="training epoch of pretraining")

    # Hyperparameters
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use. [default: 0]')
    parser.add_argument('--cuda', action='store_true', default=True, help='GPU or CPU [default: True]')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers num [default: 8]')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Learning rate decay [default: 1e-6]')
    parser.add_argument('--rn_win', type=int, default=1)

    args = parser.parse_args()
    seed_torch(args.seed)
    params = run_params(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print("params {}".format(params))
    if args.mode == "pretrain":
        run(params)
    elif args.mode == "stream":
        stream_run(params)
    else:
        raise NotImplementedError()