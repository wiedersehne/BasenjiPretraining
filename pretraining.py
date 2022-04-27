from random import randint, shuffle
from utils import *
import pandas as pd
import torch.nn as nn
import models
from torch.utils.data import Dataset, DataLoader
import train
from random import random as rand
import fire
from vcf_encode import *
import wandb
from models import LeiModel_sig_avg as LeiModel
import argparse
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

device = "cuda"
torch.backends.cudnn.bechmark = True
wandb_config = {
    "mask_ratio": 0.20,
    "real_mask": 0.80,
    "hdim": 128
}
wandb.init(config=wandb_config)



class DatasetCreator(Dataset):
    """
    Class to construct a dataset for training/inference
    """

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.data[index]
        return X

    def __len__(self):
        return len(self.data)


class SeqDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, X, batch_size, masking):
        super().__init__()
        self.masking = masking
        self.batch_size = batch_size
        self.X = X

    def get_data(self):  # iterator to load data
        corpus = []
        for i in range(len(self.X)):
            instance = self.masking(self.X[i])
            corpus.append(instance)
        # To Tensor
        print(len(corpus))
        loader = DataLoader(
                DatasetCreator(corpus),
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=True
            )

        return loader



class RandomMasking():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, mask_prob, real_mask_ratio):
        super().__init__()
        self.mask_prob = mask_prob
        self.real_mask_ratio = real_mask_ratio
        self.random_mask_ratio = 0.5

    def __call__(self, instance):
        instance = instance.transpose(1, 0)
        cand_pos = [i for i, token in enumerate(instance)]
        # For masked Language Models
        n_pred = int(self.mask_prob*len(instance))
        masked_tokens, masked_pos = [], []
        # candidate positions of masked tokens
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            x = instance[pos]
            masked_pos.append(pos)
            indx = [index for index,value in enumerate(x) if value == 1]
            if indx:
                masked_tokens.append(indx[0])
            else:
                masked_tokens.append(4)

            if rand() < self.real_mask_ratio:
                instance[pos] = torch.tensor([0, 0, 0, 0])

            elif rand() < self.random_mask_ratio:
                temp = [0, 0, 0, 0]
                indx = get_random_word()
                temp[indx] = 1
                instance[pos] = torch.tensor(temp)


        instance = torch.Tensor(instance).to(torch.int64)
        masked_tokens = torch.Tensor(masked_tokens).to(torch.int64)
        #print(masked_tokens)
        masked_pos = torch.Tensor(masked_pos).to(torch.int64)

        return (instance, masked_tokens, masked_pos)


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg, wandb_config):
        super().__init__()
        self.hdim = wandb_config.hdim
        self.norm = nn.LayerNorm(self.hdim)
        self.linear = nn.Linear(self.hdim, self.hdim)
        self.cdilNet = models.CDIL_Conv(cfg.dim, [self.hdim]*9, ks=3)
        self.activ2 = models.gelu
        self.decoder = nn.Linear(self.hdim, 5, bias=False)

    def forward(self, input_seq, masked_pos):
        input_seq = input_seq.float()
        h = torch.permute(input_seq, (0, 2, 1))
        h = self.cdilNet(h)
        h = torch.permute(h, (0, 2, 1))
        # MLM
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked)

        return logits_lm


# val and test
def net_eval(net, criterion, ep, name, eval_loader, best_losses, best_roc, out):
    OUT = out
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
    return best_losses, best_roc


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def classify(config=None,
             length = 1000,
             layer = 9,
             kernel_size = 3,
             learning_rate = 0.03,
             batch_size = 64,
             epoch = 30,
             device=device):

    # setting
    KS = kernel_size
    IN_dim = 4
    OUT = 49

    LAYER = layer
    DIM = config.hdim
    LR = learning_rate
    BATCH = batch_size
    EPOCH = epoch


    # model
    net = LeiModel(IN_dim, config.hdim, LAYER, KS, OUT)
    #load(net, f'./exp/pretrain/model_ve_1hot_steps_1000_mask_rate_0.34950581186640817_mask_ratio_0.5080529261634422_hdim_128.pt')
    load(net, f'./exp/pretrain/model_ve_1hot_steps_120000_mask_rate_{config.mask_ratio}_mask_ratio_{config.real_mask}_hdim_{config.hdim}.pt')
    net = net.to(device)
    print(net)

    PARA = sum(p.numel() for p in net.parameters() if p.requires_grad)
    p_c = sum(p.numel() for p in net.cdilNet.parameters() if p.requires_grad)
    p_l = sum(p.numel() for p in net.classifier.parameters() if p.requires_grad)
    p_all = p_c + p_l


    # optim
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=1e-6, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    best_val_losses = 999999.
    best_test_losses = 999999.
    best_val_roc = 0.
    best_test_roc = 0.


    # dataset
    VCF_train = pd.read_csv('./data/label811_49_train.csv')
    train_loader = DataLoader(vcf_Dataset(VCF_train, length), batch_size=BATCH, shuffle=True, drop_last=False, pin_memory=True)
    VCF_val = pd.read_csv('./data/label811_49_valid.csv')
    val_loader = DataLoader(vcf_Dataset(VCF_val, length), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)
    VCF_test = pd.read_csv('./data/label811_49_test.csv')
    test_loader = DataLoader(vcf_Dataset(VCF_test, length), batch_size=BATCH, shuffle=False, drop_last=False, pin_memory=True)


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

        save('./exp/pretrain', net, epoch, config.mask_ratio, config.real_mask, config.hdim)

        t_end = datetime.now()
        epoch_time = (t_end - t_start).total_seconds()
        print(f'train_loss_{train_losses_mean}_train_roc_{train_roc}_time_used_{epoch_time}')
        
        with torch.no_grad():
            net.eval()
            best_val_losses, best_val_roc = net_eval(net, criterion, epoch, 'val', val_loader, best_val_losses, best_val_roc, OUT)
            best_test_losses, best_test_roc = net_eval(net, criterion, epoch, 'test', test_loader, best_test_losses, best_test_roc, OUT)
            print(f'val_loss_{best_val_losses}_val_roc_{best_val_roc}')
            print(f'test_loss_{best_test_losses}_test_roc_{best_test_roc}')




def training(train_cfg='config/pretrain.json',
            model_cfg='config/model.json',
            data_parallel=False,
            save_dir='./exp/pretrain/',
            max_len=1000,
            mask_prob=0.20):


    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)
    config = wandb.config

    set_seeds(cfg.seed)
    # load data
    vcf_train = pd.read_csv('./data/label811_49_train.csv')
    train_data = vcf_many(vcf_train, max_len)
    # print(train_data)

    vcf_test = pd.read_csv('./data/label811_49_test.csv')
    test_data = vcf_many(vcf_test, max_len)

    vcf_val = pd.read_csv('./data/label811_49_valid.csv')
    val_data = vcf_many(vcf_val, max_len)

    masking = RandomMasking(config.mask_ratio, config.real_mask)

    train_data_loader = SeqDataLoader(train_data[0],
                                      cfg.batch_size,
                                      masking=masking)

    valid_data_loader = SeqDataLoader(val_data[0],
                                      cfg.batch_size,
                                      masking=masking)

    test_data_loader = SeqDataLoader(test_data[0],
                                     cfg.batch_size,
                                     masking=masking)

    train_data_iter = train_data_loader.get_data()
    valid_data_iter = valid_data_loader.get_data()
    test_data_iter = test_data_loader.get_data()

    model = BertModel4Pretrain(model_cfg, config)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, "sgd", cfg.lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    #                                                 max_lr = cfg.lr,
    #                                                 steps_per_epoch = int(cfg.total_steps/cfg.n_epochs),
    #                                                 epochs=cfg.n_epochs)
    # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=22470)
    trainer = train.Trainer(cfg, model, train_data_iter, valid_data_iter, test_data_iter,
                            optimizer, criterion, save_dir, device, scheduler, config)
    trainer.train(data_parallel)

    classify(config = config,
             length = max_len,
             layer = 9,
             kernel_size = 3,
             learning_rate = 0.03,
             batch_size = 64,
             epoch = 50,
             device = device)


if __name__ == '__main__':
    fire.Fire(training)
    # classify(config = wandb.config,
    #          length = 1000,
    #          layer = 9,
    #          kernel_size = 3,
    #          learning_rate = 0.03,
    #          batch_size = 64,
    #          epoch = 30,
    #          device = device)
