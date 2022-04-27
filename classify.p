import os
# import sys
import wandb
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import *
from models import LeiModel_sig_avg as LeiModel
from vcf_encode import vcf_Dataset
wandb.init(project="variant_classification", entity="tonyu")

parser = argparse.ArgumentParser(description="experiment")
parser.add_argument("--length", type=int, default=1000)
parser.add_argument("--layer", type=int, default=9)
parser.add_argument("--dim", type=int, default=128)
parser.add_argument('--lr', type=float, default=0.03)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument('--epoch', type=int, default=50)
args = parser.parse_args()


# seed
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


# setting
device = 'cuda:0'
KS = 3
IN_dim = 4
OUT = 49

LENGTH = args.length
LAYER = args.layer
DIM = args.dim
LR = args.lr
BATCH = args.batch
EPOCH = args.epoch


# model
net = LeiModel(IN_dim, DIM, LAYER, KS, OUT)
load(net, './exp/pretrain/model_ve_1hot_steps_120000_mask_rate_0.3645174385744928_mask_ratio_0.7392219304719084_hdim_128.pt')
load_cdil_head(net, './exp/pretrain/classify_ftcnn_freeze_1hot_steps_0_mask_rate_0.3645_mask_ratio_0.7392_hdim_128.pt')
load_linear_head(net, './exp/pretrain/classify_ftcnn_freeze_1hot_steps_0_mask_rate_0.3645_mask_ratio_0.7392_hdim_128.pt')
net = net.to(device)
print(net)

PARA = sum(p.numel() for p in net.parameters() if p.requires_grad)
p_c = sum(p.numel() for p in net.cdilNet.parameters() if p.requires_grad)
p_l = sum(p.numel() for p in net.classifier.parameters() if p.requires_grad)
p_all = p_c + p_l


# optim
criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=1e-6, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
best_val_losses = 999999.
best_test_losses = 999999.
best_val_roc = 0.
best_test_roc = 0.


# save
setting = 'len' + str(LENGTH) + '_L' + str(LAYER) + '_D' + str(DIM)
save_dir = 'vcf811_49_avg'
os.makedirs(save_dir,  exist_ok=True)
log_file_name = save_dir + '/' + setting + '_P' + str(PARA) + '.txt'
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
loginf = logging.info

loginf(torch.cuda.get_device_name(device))
loginf(setting)
loginf(log_file_name)
loginf('all:{}\t conv: {}/{:f}\t linear:{}/{:f}'.format(p_all, p_c, p_c/p_all, p_l, p_l/p_all))



# dataset
VCF_train = pd.read_csv('./data/label811_49_train.csv')
train_loader = DataLoader(vcf_Dataset(VCF_train, LENGTH), batch_size=BATCH, shuffle=True, drop_last=False, pin_memory=True)
VCF_val = pd.read_csv('./data/label811_49_valid.csv')
val_loader = DataLoader(vcf_Dataset(VCF_val, LENGTH), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)
VCF_test = pd.read_csv('./data/label811_49_test.csv')
test_loader = DataLoader(vcf_Dataset(VCF_test, LENGTH), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)


# val and test
def net_eval(ep, name, eval_loader, best_losses, best_roc):
    eval_losses = 0
    eval_preds = [[] for _ in range(OUT)]
    eval_labels = [[] for _ in range(OUT)]
    eval_rocs = []
    eval_aps = []

    eval_start = datetime.now()
    for eval_ref, eval_alt, eval_tissue, eval_label in eval_loader:
        eval_ref, eval_alt, eval_tissue, eval_label = eval_ref.to(device), eval_alt.to(device), eval_tissue.to(device), eval_label.to(device)
        eval_pred = net(eval_ref, eval_alt, eval_tissue)
        eval_loss = criterion(eval_pred, eval_label)
        eval_losses += eval_loss

        for t, y, l in zip(eval_tissue, eval_pred, eval_label):
            eval_preds[t.item()].append(y.item())
            eval_labels[t.item()].append(l.item())

    for eval_i in range(OUT):
        eval_rocs.append(roc_auc_score(eval_labels[eval_i], eval_preds[eval_i]))
        eval_aps.append(average_precision_score(eval_labels[eval_i], eval_preds[eval_i]))

    eval_roc = np.average(eval_rocs)
    eval_ap = np.average(eval_aps)
    eval_losses_mean = eval_losses / len(eval_loader)

    wandb.log({name + " loss": eval_losses_mean})
    wandb.log({name + " roc": eval_roc})
    # wandb.log({name + " ap": eval_ap})

    eval_end = datetime.now()
    eval_time = (eval_end - eval_start).total_seconds()

    if eval_losses_mean > best_losses:
        print_loss = '{} loss'
    else:
        best_losses = eval_losses_mean
        print_loss = 'best {} loss'
    if eval_roc < best_roc:
        print_roc = '{} roc'
    else:
        best_roc = eval_roc
        print_roc = 'best {} roc'
    print_epoch = 'Epoch {}, ' + print_loss + ': {}, task: {}, ' + print_roc + ': {}, ap: {}, Time: {}'
    loginf(print_epoch.format(ep, name, eval_losses_mean, len(eval_rocs), name, eval_roc, eval_ap, eval_time))
    return best_losses, best_roc


# train
for epoch in range(EPOCH):
    net.train()
    train_losses = 0
    train_preds = [[] for _ in range(OUT)]
    train_labels = [[] for _ in range(OUT)]
    train_rocs = []
    train_aps = []
    t_start = datetime.now()
    for ref, alt, tissue, label in train_loader:
        ref, alt, tissue, label = ref.to(device), alt.to(device), tissue.to(device), label.to(device)
        optimizer.zero_grad()
        pred = net(ref, alt, tissue)
        batch_loss = criterion(pred, label)
        batch_loss.backward()
        optimizer.step()
        train_losses += batch_loss

        for one_tissue, one_y, one_label in zip(tissue, pred, label):
            train_preds[one_tissue.item()].append(one_y.item())
            train_labels[one_tissue.item()].append(one_label.item())

    for i in range(OUT):
        train_rocs.append(roc_auc_score(train_labels[i], train_preds[i]))
        train_aps.append(average_precision_score(train_labels[i], train_preds[i]))

    train_roc = np.average(train_rocs)
    train_ap = np.average(train_aps)
    train_losses_mean = train_losses / len(train_loader)

    wandb.log({"train loss": train_losses_mean})
    wandb.log({"train roc": train_roc})
    # wandb.log({"train ap": train_ap})

    t_end = datetime.now()
    epoch_time = (t_end - t_start).total_seconds()
    # loginf('Epoch {}, train loss: {}, Time: {}'.format(epoch, train_losses_mean, epoch_time))
    loginf('Epoch {}, train loss: {}, train roc: {}, train ap: {}, Time: {}'.format(epoch, train_losses_mean, train_roc, train_ap, epoch_time))
    #save('./exp/pretrain', net, epoch, 0.3645, 0.7392, 128)

    with torch.no_grad():
        net.eval()
        best_val_losses, best_val_roc = net_eval(epoch, 'val', val_loader, best_val_losses, best_val_roc)
        best_test_losses, best_test_roc = net_eval(epoch, 'test', test_loader, best_test_losses, best_test_roc)
        loginf('=' * 100)
