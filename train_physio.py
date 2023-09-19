# This file is used to train the CfC model on the Physionet dataset.
# The code is adapted from the CfC codebase, which can be found here:
# https://github.com/raminmh/CfC/blob/master/train_physio.py
# The original CfC paper can be found here:
# https://arxiv.org/abs/2006.16972
import argparse
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.multiprocessing as mp
from torchmetrics import F1Score
from torchmetrics.functional import accuracy, auroc, average_precision
from torch_cfc import Cfc
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional
from sklearn.metrics import roc_auc_score
import sys
from pytorch_lightning.loggers import CSVLogger
from duv_physionet import get_physio
import numpy as np
import time
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from pytorch_lightning.callbacks import Callback


class SpeedCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        # The reproducing the mTAN times and calibrating to our GPU shows that my GPU is 1.33 times faster
        print(f"Took {1.34*(time.time()-self._start)/60:0.3f} minutes")


class PhysionetLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.test_step_outputs = [[], []]  # two dataloaders
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.Tensor((1.0, hparams["class_weight"]))
        )
        self._hparams = hparams
        self._all_rocs = []
        self._all_prs = []

    def _prepare_batch(self, batch):
        x, tt, mask, y = batch
        t_elapsed = tt[:, 1:] - tt[:, :-1]
        t_fill = torch.zeros(tt.size(0), 1, device=x.device)
        t = torch.cat((t_fill, t_elapsed), dim=1)
        return x, t, mask, y

    def training_step(self, batch, batch_idx):
        x, tt, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat.detach(), dim=-1)
        acc = accuracy(preds, y, task="binary")
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, tt, mask, y = self._prepare_batch(batch)

        y_hat = self.model.forward(x, tt, mask)
        y_hat = y_hat.view(-1, y_hat.size(-1))
        y = y.view(-1).long()

        loss = self.loss_fn(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        #self.test_step_outputs[dataloader_idx].append([preds, y])
        acc = accuracy(preds, y, task="binary")
        softmax = torch.nn.functional.softmax(y_hat, dim=1)[:, 1]
        self.test_step_outputs[dataloader_idx].append([softmax, y])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        f1 = F1Score(task="binary", num_classes=2)
        f1_score = f1(preds.cpu(),y.cpu())
        self.log("val_f1", f1_score, prog_bar=True)
        return [softmax, y]

    def on_validation_epoch_end(self):
        all_preds = torch.cat([l[0] for l in self.test_step_outputs[0]])
        all_labels = torch.cat([l[1] for l in self.test_step_outputs[0]])
        auprc = average_precision(all_preds.float(), all_labels, task="binary")
        self._all_prs.append(auprc)
        self.log("val_aucpr", auprc, prog_bar=True)
        auc = auroc(all_preds.float(), all_labels, task="binary")
        # f1 = BinaryF1Score(all_preds.float(), all_labels)
        # self.log("val_f1", f1, prog_bar=True)
        self._all_rocs.append(auc)
        self.log("val_rocauc", auc, prog_bar=True)


    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        optim = "rmsprop"
        if "optim" in self._hparams.keys():
            optim = self._hparams["optim"]
        optimizer = {
            "adamw": torch.optim.AdamW,
            "adam": torch.optim.Adam,
            "rmsprop": torch.optim.RMSprop,
        }[optim]
        # optimizer = torch.optim.Adam(
        optimizer = optimizer(
            self.model.parameters(),
            lr=self._hparams["base_lr"],
            weight_decay=self._hparams["weight_decay"],
        )

        def lamb_f(epoch):
            lr = self._hparams["decay_lr"] ** epoch
            # print(f"LEARNING RATE = {lr:0.4g} (epoch={epoch})")
            return lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lamb_f)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer, lambda epoch: self._hparams["decay_lr"] ** epoch
        # )
        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        closure
    ):
        optimizer.step(closure=closure)
        # Apply weight constraints
        if self._hparams["use_ltc"]:
            self.model.rnn_cell.apply_weight_constraints()


def eval(hparams, speed=False):
    # torch.set_num_threads(4)
    model = Cfc(
        in_features= 51, #54 with DR
        hidden_size=hparams["hidden_size"],
        out_feature=2,
        hparams=hparams,
        use_mixed=hparams["use_mixed"],
        use_ltc=hparams["use_ltc"],
    )
    learner = PhysionetLearner(model, hparams)

    class FakeArg:
        batch_size = 32
        classif = True
        n = 428
        extrap = False
        sample_tp = None
        cut_tp = None

    fake_arg = FakeArg()
    fake_arg.batch_size = hparams["batch_size"]
    device = "cpu"
    data_obj = get_physio(fake_arg, device)
    train_loader = data_obj["train_dataloader"]
    test_loader = data_obj["test_dataloader"]
    gpu_name = "cpu"
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpu_name = str(os.environ["CUDA_VISIBLE_DEVICES"])

    trainer = pl.Trainer(
        max_epochs=hparams["epochs"],
        gradient_clip_val=hparams["clipnorm"],
        devices = 1, accelerator = "gpu",
        callbacks=[SpeedCallback()] if speed else None,
    )
    trainer.fit(
        learner,
        train_loader,
    )
    results = trainer.test(learner, test_loader)[0]
    all_preds = torch.cat([l[0] for l in learner.test_step_outputs[0]])
    all_labels = torch.cat([l[1] for l in learner.test_step_outputs[0]])
    
    fpr, tpr, thresholds = roc_curve(all_labels.cpu(), all_preds.cpu())
    precision, recall, _ = precision_recall_curve(all_labels.cpu(), all_preds.cpu())
    
    return float(results["val_rocauc"]), float(results["val_aucpr"]), fpr, tpr, thresholds, precision, recall



# AUC: 83.90 % +-0.22

# # AUC: 88.37 % +-0.87
BEST_DEFAULT = {
    "epochs": 34, # also try out different epochs or just use an Early Stopping mechanism (latter is better)
    "class_weight": 0.25,
    "clipnorm": 0,
    "hidden_size": 512,
    "base_lr": 0.002,
    "decay_lr": 0.9,
    "backbone_activation": "lecun",
    "backbone_units": 64,
    "backbone_dr": 0.2,
    "backbone_layers": 1,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.5, #try 53 55 and 0.6  
    "batch_size": 16, #maybe try 32? idk. try out different batch sizes, its a hyperparameter
    "use_mixed": False, #this is an intriguing case, seems to decrease model perf variance but not sure at all
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}



# Ignore the rest for now for the lifespan prediction problem
# 0.8397588133811951
BEST_MIXED = {
    "epochs": 20,
    "class_weight": 0.71,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.001,
    "decay_lr": 0.9,
    "backbone_activation": "lecun",
    "backbone_units": 64,
    "backbone_dr": 0.3,
    "backbone_layers": 2,
    "weight_decay": 4e-06,
    "optim": "adamw",
    "init": 0.6,
    "batch_size": 32,
    "use_mixed": True,
    "no_gate": False,
    "minimal": False,
    "use_ltc": False,
}

# 0.8395 $\pm$ 0.0033
BEST_NO_GATE = {
    "epochs": 30,
    "class_weight": 0.32,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.73,
    "backbone_activation": "relu",
    "backbone_units": 192,
    "backbone_dr": 0.0,
    "backbone_layers": 2,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.55,
    "batch_size": 8,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
    "use_ltc": False,
}
# test AUC 0.6431 $\pm$ 0.0180
BEST_MINIMAL = {
    "epochs": 116,
    "class_weight": 18.25,
    "clipnorm": 0,
    "hidden_size": 64,
    "base_lr": 0.003,
    "decay_lr": 0.72,
    "backbone_activation": "tanh",
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 5e-05,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 128,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
    "use_ltc": False,
}
# 0.6577
BEST_LTC = {
    "optimizer": "adam",
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "forget_bias": 2.4,
    "epochs": 80,
    "class_weight": 8,
    "clipnorm": 0,
    "hidden_size": 64,
    "backbone_units": 64,
    "backbone_dr": 0.1,
    "backbone_layers": 3,
    "weight_decay": 0,
    "optim": "adamw",
    "init": 0.53,
    "batch_size": 64,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}


def score(config, n=5):

    means = []
    means2 = []
    fpr_list = []
    tpr_list = []


    for i in range(n):
        a, b, fpr, tpr, _, precision,recall = eval(config, speed=True)
        means.append(a)
        means2.append(b)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    print(f"Test ROCAUC: {np.mean(means):0.4f} $\\pm$ {np.std(means):0.4f} ")
    print(f"Test AUCPR: {np.mean(means2):0.4f} $\\pm$ {np.std(means2):0.4f} ")
    return fpr_list, tpr_list, precision, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_mixed", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--use_ltc", action="store_true")
    args = parser.parse_args()


    if args.minimal:
        fpr_list, tpr_list, precision, recall = score(BEST_MINIMAL)
    elif args.no_gate:
        fpr_list, tpr_list, precision, recall = score(BEST_NO_GATE)
    elif args.use_ltc:
        fpr_list, tpr_list, precision, recall = score(BEST_LTC)
    elif args.use_mixed:
        fpr_list, tpr_list, precision, recall = score(BEST_MIXED)
    else:
        fpr_list, tpr_list, precision, recall = score(BEST_DEFAULT)

    ## Set the parameter 'n' of score function to 1 in case of plotting 
    # Plot ROC curve
    # plt.figure()
    # for i in range(len(fpr_list)):
    #     plt.plot(fpr_list[i], tpr_list[i], label=f'ROC Curve {i+1}')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend()
    # plt.savefig('roc_curve.png')

    # # Close the figure to release resources (optional)
    # plt.close()
    # pr_auc = auc(recall, precision)

    # # Plot Precision-Recall curve
    # plt.figure()
    # plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR Curve (AUC = {pr_auc:.2f})')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.title('Precision-Recall Curve')
    # plt.legend()

    # # Save the Precision-Recall curve as an image file (e.g., PNG)
    # plt.savefig('precision_recall_curve.png')

    # # Close the figure to release resources (optional)
    # plt.close()