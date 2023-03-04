import math
import gc

import cv2
import numpy as np
import pandas as pd
import pydicom
import pytorch_lightning as pl
import sklearn.metrics
import timm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler
from PIL import Image
import albumentations as A

from tqdm import tqdm

def pfbeta(labels, predictions, beta=1):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if labels[idx]:
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-7)
    c_recall = ctp / (y_true_count + 1e-7)
    if c_precision > 0 and c_recall > 0:
        result = (
            (1 + beta_squared)
            * (c_precision * c_recall)
            / (beta_squared * c_precision + c_recall + 1e-7)
        )
        return result
    else:
        return 0

def optimal_f1(labels, predictions):
    thres = np.arange(0, 1, 0.01)
    f1s = [pfbeta(labels, predictions > thr) for thr in thres]
    idx = np.argmax(f1s)
    return f1s[idx], thres[idx]
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False, pos_weight=self.pos_weight)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float,
        m: float,
        easy_margin: bool,
        ls_eps: float,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input: torch.Tensor, label: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # Enable 16 bit precision
        cosine = cosine.to(torch.float32)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
    
"""
def pfbeta(labels, preds, beta=1):
    eps = 1e-5
    preds = preds.clip(0, 1)
    y_true_count = labels.sum()
    ctp = preds[labels==1].sum()
    cfp = preds[labels==0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + 1e-7)
    c_recall = ctp / (y_true_count + 1e-7)
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + 1e-7)
        return result
    else:
        return 0.0
"""

class CNNModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg["arcface"] is None or cfg["task"]["aux_target"] is None:
            num_classes = 1
        else:
            num_classes = None
        self.model = timm.create_model(
            model_name=cfg["model"]["model_name"],
            pretrained=cfg["model"]["pretrained"],
            in_chans=cfg["model"]["in_chans"],
            num_classes=num_classes,
            drop_rate=cfg["model"]["drop_rate"],
            drop_path_rate=cfg["model"]["drop_path_rate"],
        )
        
        if cfg["model"]["criterion"] == "FocalLoss":
            self.criterion = FocalLoss(logits=True, pos_weight=cfg["model"]["loss_weights"])
        else:
            self.criterion = nn.__dict__[cfg["model"]["criterion"]](pos_weight=cfg["model"]["loss_weights"])
            
        if cfg["arcface"] is not None:
            self.embedding = nn.Linear(self.model.get_classifier().in_features, cfg["arcface"]["params"]["in_features"])
            self.arc = ArcMarginProduct(**cfg["arcface"]["params"])
            self.criterion = nn.CrossEntropyLoss()
            self.model.reset_classifier(num_classes=0, global_pool="avg")
        elif cfg["task"]["aux_target"] is not None:
            self.nn_cancer = torch.nn.Sequential(torch.nn.Linear(self.model.get_classifier().in_features, 1))
            self.nn_aux = torch.nn.ModuleList([torch.nn.Linear(self.model.get_classifier().in_features, n) for n in cfg["task"]["aux_target_nclasses"]])
            self.model.reset_classifier(num_classes=0, global_pool="avg")
        
        if cfg["model"]["grad_checkpointing"]:
            print("grad_checkpointing true")
            self.model.set_grad_checkpointing(enable=True)

    def forward(self, X):
        if self.cfg["arcface"] is not None:
            X = self.model(X)
            outputs = self.embedding(X)
        elif self.cfg["task"]["aux_target"] is not None:
            X = self.model(X)
            cancer = self.nn_cancer(X) #.squeeze()
            aux = []
            for nn in self.nn_aux:
                aux.append(nn(X)) #.squeeze()
            return cancer, aux
        else:
            outputs = self.model(X)
        return outputs
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self, x, y, alpha=1.0):
        indices = torch.randperm(x.size(0))
        shuffled_data = x[indices]
        shuffled_target = y[indices]

        lam = np.clip(np.random.beta(alpha, alpha),0.3,0.4)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        new_data = x.clone()
        new_data[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return new_data, y, shuffled_target, lam
    
    def mixup_data(self, x, y, alpha=1.0, return_index=False):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        if return_index:
            return mixed_x, y_a, y_b, lam, index
        else:
            return mixed_x, y_a, y_b, lam
    
    def mix_criterion(self, pred, y_a, y_b, lam, criterion="default"):
        if criterion == "default":
            criterion = self.criterion
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def training_step(self, batch, batch_idx):
        if self.cfg["model"]["train_2nd"] and self.current_epoch >= (self.cfg["pl_params"]["max_epochs"] - self.cfg["model"]["epoch_2nd"]):
            # 最後だけaugmentation切るとかする用
            self.cfg["model"]["aug_mix"] = False      
        if self.cfg["model"]["aug_mix"] and torch.rand(1) < 0.5:
            if self.cfg["arcface"] is not None:
                X, y = batch
                mixed_X, y_a, y_b, lam = self.mixup_data(X, y)
                pred_y = self.forward(mixed_X)
                pred_y = self.arc(pred_y, y, self.device)
                loss = self.mix_criterion(pred_y ,y_a.long().squeeze(), y_b.long().squeeze(), lam)
                self.log("train_loss", loss, prog_bar=True)
                return loss
            elif self.cfg["task"]["aux_target"] is not None:
                X, y, aux_y = batch
                mixed_X, y_a, y_b, lam, index = self.mixup_data(X, y, return_index=True)
                pred_y, pred_aux_y = self.forward(mixed_X)
                aux_y_a, aux_y_b = aux_y, aux_y[index]
                cancer_loss = self.mix_criterion(pred_y ,y_a, y_b, lam)
                aux_loss = torch.mean(torch.stack([self.mix_criterion(pred_aux_y[i], aux_y_a[:, i], aux_y_b[:, i], lam, criterion=torch.nn.functional.cross_entropy) for i in range(aux_y.shape[-1])]))
                loss = cancer_loss + self.cfg["task"]["aux_loss_weight"] * aux_loss
                self.log("cancer_loss", cancer_loss, prog_bar=False)
                self.log("aux_loss", aux_loss, prog_bar=False)
                self.log("train_loss", loss, prog_bar=True)
                return {"loss": loss, "cancer_loss": cancer_loss, "aux_loss": aux_loss}
            else:
                X, y = batch
                #if torch.rand(1) >= 0.5:
                #    mixed_X, y_a, y_b, lam = self.mixup_data(X, y)
                #else:
                #    mixed_X, y_a, y_b, lam = self.cutmix_data(X, y)
                mixed_X, y_a, y_b, lam = self.mixup_data(X, y)
                pred_y = self.forward(mixed_X)
                loss = self.mix_criterion(pred_y ,y_a, y_b, lam)
                self.log("train_loss", loss, prog_bar=True)
                return loss
        else:
            if self.cfg["arcface"] is not None:
                X, y = batch
                pred_y = self.arc(pred_y, y, self.device)
                loss = self.criterion(pred_y, y.long().squeeze())
                self.log("train_loss", loss, prog_bar=True)
                return loss
            elif self.cfg["task"]["aux_target"] is not None:
                X, y, aux_y = batch
                pred_y, pred_aux_y = self.forward(X)
                cancer_loss = self.criterion(pred_y, y)
                aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(pred_aux_y[i], aux_y[:, i]) for i in range(aux_y.shape[-1])]))
                loss = cancer_loss + self.cfg["task"]["aux_loss_weight"] * aux_loss
                self.log("cancer_loss", cancer_loss, prog_bar=False)
                self.log("aux_loss", aux_loss, prog_bar=False)
                self.log("train_loss", loss, prog_bar=True)
                return {"loss": loss, "cancer_loss": cancer_loss, "aux_loss": aux_loss}
            else:
                X, y = batch
                pred_y = self.forward(X)
                loss = self.criterion(pred_y, y)
                self.log("train_loss", loss, prog_bar=True)
                return loss

    def training_epoch_end(self, outputs):
        loss_list = [x["loss"] for x in outputs]
        avg_loss = torch.stack(loss_list).mean()
        self.log("train_avg_loss", avg_loss, prog_bar=True)
        
        if self.cfg["task"]["aux_target"] is not None:
            cancer_loss_list = [x["cancer_loss"] for x in outputs]
            aux_loss_list = [x["aux_loss"] for x in outputs]
            cancer_avg_loss = torch.stack(cancer_loss_list).mean()
            aux_avg_loss = torch.stack(aux_loss_list).mean()
            self.log("train_avg_cancer_loss", cancer_avg_loss, prog_bar=False)
            self.log("train_avg_aux_loss", aux_avg_loss, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        if self.cfg["arcface"] is not None:
            X, y = batch
            pred_y = self.forward(X)
            pred_y = self.arc(pred_y, y, self.device)
            loss = self.criterion(pred_y, y.long().squeeze())
            pred_y = nn.Softmax(dim=1)(pred_y)
            pred_y = pred_y[:, 1]
            pred_y = torch.nan_to_num(pred_y)
            return {"valid_loss": loss, "preds": pred_y, "targets": y}
        elif self.cfg["task"]["aux_target"] is not None:
            X, y, aux_y = batch
            pred_y, pred_aux_y = self.forward(X)
            cancer_loss = self.criterion(pred_y, y)
            aux_loss = torch.mean(torch.stack([torch.nn.functional.cross_entropy(pred_aux_y[i], aux_y[:, i]) for i in range(aux_y.shape[-1])]))
            loss = cancer_loss + self.cfg["task"]["aux_loss_weight"] * aux_loss
            pred_y = torch.sigmoid(pred_y)
            pred_y = torch.nan_to_num(pred_y)
            return {"valid_loss": loss, "cancer_loss": cancer_loss, "aux_loss": aux_loss, "preds": pred_y, "targets": y}
        else:
            X, y = batch
            pred_y = self.forward(X)
            loss = self.criterion(pred_y, y)
            pred_y = torch.sigmoid(pred_y)
            pred_y = torch.nan_to_num(pred_y)
            return {"valid_loss": loss, "preds": pred_y, "targets": y}

    def validation_epoch_end(self, outputs):
        loss_list = [x["valid_loss"] for x in outputs]
        preds = torch.cat([x["preds"] for x in outputs], dim=0).cpu().detach().numpy()
        targets = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        avg_loss = torch.stack(loss_list).mean()
        pfbeta_score = pfbeta(targets.flatten(), preds.flatten())
        if np.unique(targets).shape[0] == 1:
            auc_score = 0.0
        else:
            auc_score = sklearn.metrics.roc_auc_score(targets.flatten(), preds.flatten())
        optimized_pfbeta_score, threshold = optimal_f1(targets.flatten(), preds.flatten())
        recall = sklearn.metrics.recall_score(targets.flatten(), preds.flatten() > threshold)
        specificity = sklearn.metrics.recall_score(targets.flatten(), preds.flatten() > threshold, pos_label=0)
        precision = sklearn.metrics.precision_score(targets.flatten(), preds.flatten() > threshold)
        self.log("valid_avg_loss", avg_loss, prog_bar=True)
        self.log("valid_pfbeta_score", pfbeta_score, prog_bar=True)
        self.log("optimized_pfbeta_score", optimized_pfbeta_score, prog_bar=True)
        self.log("valid_auc_score", auc_score, prog_bar=True)
        self.log("threshold", threshold, prog_bar=False)
        self.log("recall", recall, prog_bar=False)
        self.log("specificity", specificity, prog_bar=False)
        self.log("precision", precision, prog_bar=False)
        
        if self.cfg["task"]["aux_target"] is not None:
            cancer_loss_list = [x["cancer_loss"] for x in outputs]
            aux_loss_list = [x["aux_loss"] for x in outputs]
            cancer_avg_loss = torch.stack(cancer_loss_list).mean()
            aux_avg_loss = torch.stack(aux_loss_list).mean()
            self.log("valid_avg_cancer_loss", cancer_avg_loss, prog_bar=False)
            self.log("valid_avg_aux_loss", aux_avg_loss, prog_bar=False)
        
        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self.cfg["tta"] == "flip":
            X, X_2, y = batch
            
            if self.cfg["arcface"] is not None:
                pred_y_1 = self.forward(X)
                pred_y_2 = self.forward(X_2)
                pred_y_1 = self.arc(pred_y_1, y, self.device)      
                pred_y_2 = self.arc(pred_y_2, y, self.device) 
                pred_y_1 = nn.Softmax(dim=1)(pred_y_1)
                pred_y_2 = nn.Softmax(dim=1)(pred_y_2)
                pred_y_1 = pred_y_1[:, 1]
                pred_y_2 = pred_y_2[:, 1]
            elif self.cfg["task"]["aux_target"] is not None:
                pred_y_1, _ = self.forward(X)
                pred_y_2, _ = self.forward(X_2)
                pred_y_1 = torch.sigmoid(pred_y_1)
                pred_y_2 = torch.sigmoid(pred_y_2)
            else:
                pred_y_1 = self.forward(X)
                pred_y_2 = self.forward(X_2)
                pred_y_1 = torch.sigmoid(pred_y_1)
                pred_y_2 = torch.sigmoid(pred_y_2)
                
            pred_y = (pred_y_1 + pred_y_2) / 2.0
        else:
            X, y = batch
            
            if self.cfg["arcface"] is not None:
                pred_y = self.forward(X)
                pred_y = self.arc(pred_y, y, self.device)        
                pred_y = nn.Softmax(dim=1)(pred_y)
                pred_y = pred_y[:, 1]
            elif self.cfg["task"]["aux_target"] is not None:
                pred_y, _ = self.forward(X)
                pred_y = torch.sigmoid(pred_y)
            else:
                pred_y = self.forward(X)
                pred_y = torch.sigmoid(pred_y)
                
        return pred_y

    def configure_optimizers(self):
        optimizer = optim.__dict__[self.cfg["model"]["optimizer"]["name"]](
            self.parameters(), **self.cfg["model"]["optimizer"]["params"]
        )
        if self.cfg["model"]["scheduler"] is None:
            return [optimizer]
        else:
            if self.cfg["model"]["scheduler"]["name"] == "OneCycleLR":
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    #steps_per_epoch=self.cfg["len_train_loader"] // self.cfg["pl_params"]["accumulate_grad_batches"],
                    total_steps=self.trainer.estimated_stepping_batches,
                    **self.cfg["model"]["scheduler"]["params"],
                )
                scheduler = {"scheduler": scheduler, "interval": "step"}
            elif self.cfg["model"]["scheduler"]["name"] == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, **self.cfg["model"]["scheduler"]["params"],
                )
                scheduler = {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "valid_avg_loss",
                }
            else:
                scheduler = optim.lr_scheduler.__dict__[
                    self.cfg["model"]["scheduler"]["name"]
                ](optimizer, **self.cfg["model"]["scheduler"]["params"])
            return [optimizer], [scheduler]


class RSNADataset(torch.utils.data.Dataset):
    def __init__(self, cfg, X, y=None, augmentation=False, cutout=False, aux=False):
        self.cfg = cfg
        self.augmentation = augmentation
        self.cutout = cutout
        self.df = X
        self.aux = aux
        
        if cfg["model"]["in_chans"] == 3:
            self.mean = cfg["model"]["mean"]
            self.std = cfg["model"]["std"]
        else:
            self.mean = cfg["model"]["mean"]
            self.std = cfg["model"]["std"]
        
        if y is None:
            self.y = torch.zeros(len(self.df), dtype=torch.float32)
        else:
            self.y = torch.tensor(y.values, dtype=torch.float32)
        
        # normalize
        self.normalize = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            torchvision.transforms.Resize((cfg["task"]["height_size"], cfg["task"]["width_size"])),
            #torchvision.transforms.Normalize(mean=self.mean, std=self.std),
            
            #torchvision.transforms.Resize(min(cfg["task"]["height_size"], cfg["task"]["width_size"])),
            #torchvision.transforms.CenterCrop((cfg["task"]["height_size"], cfg["task"]["width_size"])),
        ])
        # resize
        #self.resize = torchvision.transforms.CenterCrop((cfg["task"]["height_size"], cfg["task"]["width_size"]))
        
        
        # augmentation
        # flip
        self.aug_hori_flip = A.HorizontalFlip(p=0.5)
        self.aug_ver_flip = A.VerticalFlip(p=0.5)
        # elastic and grid
        self.aug_distortion = A.GridDistortion(p=0.5)
        """
        A.OneOf([
            A.ElasticTransform(p=0.5),
            A.GridDistortion(p=0.5)
        ], p=0.5)
        """
        # affine
        #self.aug_affine = A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.1), rotate=(-45, 45), shear=(-15, 15), p=0.8)
        self.aug_affine = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.8)
        # clahe
        self.aug_clahe = A.CLAHE(p=0.5)
        # bright
        self.aug_bright = A.OneOf([
            A.RandomGamma(gamma_limit=(50, 150), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5)
        ], p=0.5)
        # cutout
        self.aug_cutout = A.CoarseDropout(max_height=8, max_width=8, p=0.5)
        # randomcrop
        #self.randomcrop = A.RandomResizedCrop(height=cfg["task"]["height_size"], width=cfg["task"]["width_size"], scale=(0.1, 1.0), ratio=(0.5, 1.0), p=0.8)

    def __len__(self):
        return len(self.df)
    
    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_LINEAR):
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)

        return resized

    def __getitem__(self, index):
        image_id = self.df.loc[index, "image_id"]
        patient_id = self.df.loc[index, "patient_id"]
        extend = self.cfg["task"]["extend"]
        size = self.cfg["task"]["size"]
        bit = self.cfg["task"]["bit"]
        if self.cfg["task"]["voi"]:
            image_path = (
                f"{self.cfg['general']['input_path']}/{extend}_{size}_{bit}bit_voi/{patient_id}_{image_id}"
            )
        else:
            image_path = (
                f"{self.cfg['general']['input_path']}/{extend}_{size}_{bit}bit/{patient_id}_{image_id}"
            )
            
        if self.cfg["model"]["in_chans"] == 3:
            X = cv2.imread(f"{image_path}.png")
        else:
            X = cv2.imread(f"{image_path}.png", cv2.IMREAD_GRAYSCALE)
        
        """
        shape = X.shape
        if shape[0] > shape[1]:
            X = self.image_resize(X, height=self.cfg["task"]["height_size"])
        else:
            X = self.image_resize(X, width=self.cfg["task"]["width_size"])
        X = Image.fromarray(X)
        X = self.resize(X)
        X = np.array(X)
        """
    
        # augmentation
        if self.augmentation:
            X = self.aug_hori_flip(image=X)["image"]
            X = self.aug_ver_flip(image=X)["image"]
            #X = self.aug_distortion(image=X)["image"]
            #X = self.aug_clahe(image=X)["image"]
            X = self.aug_affine(image=X)["image"]
            X = self.aug_bright(image=X)["image"]
            if self.cutout:
                X = self.aug_cutout(image=X)["image"]
            #X = self.randomcrop(image=X)["image"]
            
        y = self.y[index].unsqueeze(0)
            
        if self.cfg["tta"] == "flip":
            X_2 = self.aug_hori_flip(image=X)["image"]
            X = self.normalize(X)
            X_2 = self.normalize(X_2)
            return X, X_2, y
        elif self.cfg["tta"] == "clahe":
            X = self.aug_clahe(image=X)["image"]
            X = self.normalize(X)
            return X, y
        else:
            X = self.normalize(X)
            #X = torchvision.transforms.ToTensor()(X)
            #if self.normalize:
            #    if self.cfg["model"]["normalize_method"] == "z_intensity":
            #        X = torchvision.transforms.Normalize(mean=torch.mean(X, dim=(1, 2)), std=torch.std(X, dim=(1, 2))+1e-7)(X)
            #    else:
            #        X = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
            #X = torchvision.transforms.Resize((self.cfg["task"]["height_size"], self.cfg["task"]["width_size"]))(X)
            if self.aux:
                aux_y = self.df.iloc[index][self.cfg["task"]["aux_target"]].astype(np.int64)
                aux_y = torch.tensor(aux_y.values, dtype=torch.long)
                return X, y, aux_y
            else:
                return X, y

"""
class BalancedSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, dataset, num_samples=None
    ):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        df = pd.DataFrame()
        df["label"] = dataset.y
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples
"""

class WeightedBatchOverSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, labels, batch_size, minor_weight=0.5, gradient_accumulation=1
    ):
        self.labels = labels
        self.gradient_accumulation = gradient_accumulation

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_major_indices = 0
        self.used_minor_indices = 0
        self.count = 0
        self.batch_size = batch_size
        self.minor_n_samples = int(batch_size * gradient_accumulation * minor_weight)
        self.major_n_samples = batch_size * gradient_accumulation - self.minor_n_samples

    def __iter__(self):
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        self.count = 0
        self.used_major_indices = 0
        self.used_minor_indices = 0
        while self.count + self.major_n_samples <= len(self.major_indices):
            indices = (
                self.major_indices[
                    self.used_major_indices : self.used_major_indices + self.major_n_samples
                ].tolist()
                + np.random.choice(
                    self.minor_indices, self.minor_n_samples, replace=False
                ).tolist()
            )
            np.random.shuffle(indices)

            step_start = 0
            for g in range(self.gradient_accumulation):
                yield indices[step_start:step_start+self.batch_size]
                step_start += len(indices[step_start:step_start+self.batch_size])

            self.used_major_indices += self.major_n_samples
            self.used_minor_indices += self.minor_n_samples
            self.count += self.major_n_samples

    def __len__(self):
        return (len(self.major_indices) // self.major_n_samples) * self.gradient_accumulation


"""
class WeightedBatchOverSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, labels, batch_size, minor_weight=0.5, gradient_accumulation=1
    ):
        self.labels = labels
        self.gradient_accumulation = gradient_accumulation

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_major_indices = 0
        self.used_minor_indices = 0
        self.count = 0
        self.batch_size = batch_size
        self.minor_n_samples = int(batch_size * gradient_accumulation * minor_weight)
        self.major_n_samples = batch_size * gradient_accumulation - self.minor_n_samples

    def __iter__(self):
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        self.count = 0
        self.used_major_indices = 0
        self.used_minor_indices = 0
        while self.count + self.major_n_samples <= len(self.major_indices):
            if len(self.minor_indices[self.used_minor_indices : self.used_minor_indices + self.minor_n_samples].tolist()) == 0:
                # minor classを使い果たしたらランダムチョイスでオーバーサンプリング
                indices = (
                    self.major_indices[
                        self.used_major_indices : self.used_major_indices + self.major_n_samples
                    ].tolist()
                    + np.random.choice(
                        self.minor_indices, self.minor_n_samples, replace=False
                    ).tolist()
                )
            else:
                indices = (
                    self.major_indices[
                        self.used_major_indices : self.used_major_indices + self.major_n_samples
                    ].tolist()
                    + self.minor_indices[
                        self.used_minor_indices : self.used_minor_indices + self.minor_n_samples
                    ].tolist()
                )
            np.random.shuffle(indices)
            
            step_start = 0
            for g in range(self.gradient_accumulation):
                if len(indices[step_start:step_start+self.batch_size]) == self.batch_size: # drop_last
                    yield indices[step_start:step_start+self.batch_size]
                step_start += len(indices[step_start:step_start+self.batch_size])

            self.used_major_indices += self.major_n_samples
            self.used_minor_indices += self.minor_n_samples
            self.count += self.major_n_samples

    def __len__(self):
        return (len(self.major_indices) // self.major_n_samples) * self.gradient_accumulation
"""
    
"""
class WeightedBatchOverSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, labels, batch_size, minor_weight=0.5
    ):
        self.labels = labels

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_indices = 0
        self.count = 0
        self.batch_size = batch_size
        self.minor_n_samples = int(batch_size * minor_weight)
        self.major_n_samples = batch_size - self.minor_n_samples

    def __iter__(self):
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        self.count = 0
        self.used_indices = 0
        while self.count + self.major_n_samples <= len(self.major_indices):
            indices = (
                self.major_indices[
                    self.used_indices : self.used_indices + self.major_n_samples
                ].tolist()
                + np.random.choice(
                    self.minor_indices, self.minor_n_samples, replace=False
                ).tolist()
            )
            np.random.shuffle(indices)
            yield indices
            
            self.used_indices += self.major_n_samples
            self.count += self.major_n_samples

    def __len__(self):
        return len(self.major_indices) // self.major_n_samples
"""
    
class BalancedBatchOverSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, labels, batch_size,
    ):
        self.labels = labels

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_indices = 0
        self.count = 0
        self.batch_size = batch_size
        self.n_samples = batch_size // 2

    def __iter__(self):
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples <= len(self.major_indices):
            indices = (
                self.major_indices[
                    self.used_indices : self.used_indices + self.n_samples
                ].tolist()
                + np.random.choice(
                    self.minor_indices, self.n_samples, replace=False
                ).tolist()
            )
            np.random.shuffle(indices)
            yield indices

            self.used_indices += self.n_samples
            self.count += self.n_samples

    def __len__(self):
        return len(self.major_indices) // self.n_samples


class BalancedBatchUnderSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self, labels, batch_size,
    ):
        self.labels = labels

        label_counts = np.bincount(labels)
        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(labels == major_label)[0]
        self.minor_indices = np.where(labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_indices = 0
        self.count = 0
        self.batch_size = batch_size
        self.n_samples = batch_size // 2

    def __iter__(self):
        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)
        self.count = 0
        self.used_indices = 0
        while self.count + self.n_samples <= len(self.minor_indices):
            indices = (
                self.minor_indices[
                    self.used_indices : self.used_indices + self.n_samples
                ].tolist()
                + np.random.choice(
                    self.major_indices, self.n_samples, replace=False
                ).tolist()
            )
            np.random.shuffle(indices)
            yield indices

            self.used_indices += self.n_samples
            self.count += self.n_samples

    def __len__(self):
        return len(self.minor_indices) // self.n_samples


class RSNADataModule(pl.LightningDataModule):
    def __init__(self, cfg, train_loader=None, valid_loader=None, test_loader=None, train_loader_2nd=None):
        super().__init__()
        self.cfg = cfg
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.train_loader_2nd = train_loader_2nd

    def setup(self, stage):
        pass

    def train_dataloader(self):
        if self.cfg["model"]["train_2nd"]:
            if self.trainer.current_epoch >= (self.cfg["pl_params"]["max_epochs"] - self.cfg["model"]["epoch_2nd"]):
                print("ok")
                return self.train_loader_2nd
            else:
                return self.train_loader
        else:
            return self.train_loader
    
    def val_dataloader(self):
        return self.valid_loader

    def test_dataloader(self):
        return self.test_loader


class CNNClassifier:
    def __init__(self, cfg, train_X, train_y, valid_X=None, valid_y=None):
        # aux
        if cfg["task"]["aux_target"] is not None:
            cfg["task"]["aux_target_nclasses"] = train_X[cfg["task"]["aux_target"]].max() + 1
            aux = True
        else:
            aux = False
        # create datasets
        train_dataset = RSNADataset(cfg, train_X, train_y, True, True, aux=aux)
        
        """
        # get mean and std
        # vely slow...
        if cfg["model"]["normalize_method"] == "z_normalization":
            dataloader_for_get_mean_std = torch.utils.data.DataLoader(
                RSNADataset(cfg, train_X, train_y, False, False, False), **cfg["train_loader"],
            )
            mean = 0.
            std = 0.
            nb_samples = 0.
            for data, _ in tqdm(dataloader_for_get_mean_std):
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples
            mean /= nb_samples
            std /= nb_samples
            cfg["model"]["mean"] = mean
            cfg["model"]["std"] = std
            del dataloader_for_get_mean_std
            gc.collect()
            torch.cuda.empty_cache()
            print("mean: ", mean)
            print("std: ", std)
        """
            
        # for 2nd train
        if cfg["model"]["train_2nd"]:
            train_dataset_2nd = RSNADataset(cfg, train_X, train_y, True, False, aux=aux)
            train_dataloader_2nd = torch.utils.data.DataLoader(
                train_dataset_2nd, **cfg["train_loader"],
            )
        else:
            train_dataloader_2nd = None
        
        # sampler
        if cfg["model"]["batch_balanced"] == "oversampling":
            print("batch oversampling")
            cfg["train_loader"]["batch_sampler"] = BalancedBatchOverSampler(
                    train_y.values, cfg["train_loader"]["batch_size"]
                )
            del cfg["train_loader"]["batch_size"], cfg["train_loader"]["shuffle"], cfg["train_loader"]["drop_last"]
        elif cfg["model"]["batch_balanced"] == "undersampling":
            print("batch undersampling")
            cfg["train_loader"]["batch_sampler"] = BalancedBatchUnderSampler(
                    train_y.values, cfg["train_loader"]["batch_size"]
                )
            del cfg["train_loader"]["batch_size"], cfg["train_loader"]["shuffle"], cfg["train_loader"]["drop_last"]
        elif cfg["model"]["batch_balanced"] == "weighted":
            print("weighted batch sampling")
            cfg["train_loader"]["batch_sampler"] = WeightedBatchOverSampler(
                    train_y.values, cfg["train_loader"]["batch_size"], cfg["model"]["minor_weight"], cfg["pl_params"]["accumulate_grad_batches"]
                )
            del cfg["train_loader"]["batch_size"], cfg["train_loader"]["shuffle"], cfg["train_loader"]["drop_last"]
            
        """
        elif cfg["model"]["sampler"] == "balanced":
            print("using balanced sampler")
            cfg["train_loader"]["sampler"] = BalancedSampler(train_dataset)
            del cfg["train_loader"]["shuffle"]
        elif cfg["model"]["sampler"] == "weighted":
            print("using weighted sampler")
            print("sampler weights:", cfg["model"]["sampler_weights"])
            
            labels = train_y.values.flatten()
            num_samples = len(labels)
            _, train_counts = np.unique(labels, return_counts=True)
            class_weights = [num_samples/train_counts[i]*cfg["model"]["sampler_weights"][i] for i in range(len(train_counts))]
            weights = [class_weights[labels[i]] for i in range(int(num_samples))]
            
            cfg["train_loader"]["sampler"] = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
            del cfg["train_loader"]["shuffle"]
        """
        
        # data loader
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, **cfg["train_loader"],
        )
        cfg["len_train_loader"] = len(train_dataloader)
        
        # valid
        if valid_X is None:
            valid_dataloader = None
        else:
            valid_dataset = RSNADataset(cfg, valid_X, valid_y, aux=aux)
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset, **cfg["valid_loader"],
            )
            
        self.datamodule = RSNADataModule(cfg, train_loader=train_dataloader, valid_loader=valid_dataloader, train_loader_2nd=train_dataloader_2nd)
            
        # for loss weight
        if cfg["model"]["weighted_loss"] is not None:
            if cfg["model"]["weighted_loss"] == "inverse":
                labels = train_y.values.flatten()
                _, train_counts = np.unique(labels, return_counts=True)
                loss_weights = train_counts[0] / train_counts[1]
                loss_weights = torch.tensor([loss_weights], dtype=torch.float32)
            else:
                loss_weights = torch.tensor([cfg["model"]["weighted_loss"]], dtype=torch.float32)
            print("loss weights:", loss_weights)
            cfg["model"]["loss_weights"] = loss_weights
        else:
            cfg["model"]["loss_weights"] = None

        callbacks = [pl.callbacks.LearningRateMonitor(logging_interval="step")]

        if cfg["model"]["early_stopping_patience"] is not None:
            callbacks.append(
                pl.callbacks.EarlyStopping(
                    "valid_avg_loss", patience=cfg["model"]["early_stopping_patience"],
                )
            )

        if cfg["model"]["model_save"]:
            callbacks.append(
                pl.callbacks.ModelCheckpoint(
                    dirpath=f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}",
                    filename=f"last_epoch_fold{cfg['fold_n']}"
                    if cfg["general"]["cv"]
                    else f"last_epoch",
                    save_weights_only=cfg["model"]["save_weights_only"],
                )
            )

        logger = WandbLogger(
            project=cfg["general"]["project_name"],
            name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
            group=f"{cfg['general']['save_name']}_cv"
            if cfg["general"]["cv"]
            else f"{cfg['general']['save_name']}_all",
            job_type=cfg["job_type"],
            mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
            config=cfg,
        )

        self.model = CNNModel(cfg)
        self.cfg = cfg

        self.trainer = Trainer(
            callbacks=callbacks,
            logger=logger,
            reload_dataloaders_every_n_epochs=1 if cfg["model"]["train_2nd"] else 0,
            **self.cfg["pl_params"]
        )

    def train(self, weight_path=None):
        self.trainer.fit(
            self.model,
            datamodule=self.datamodule,
            ckpt_path=weight_path,
        )

    def predict(self, test_X, weight_path=None):
        preds = []
        test_dataset = RSNADataset(self.cfg, test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()
        return preds

    def load_weight(self, weight_path):
        self.model = self.model.load_from_checkpoint(
            checkpoint_path=weight_path, cfg=self.cfg,
        )
        print(f"loaded model ({weight_path})")


class CNNClassifierInference:
    def __init__(self, cfg, weight_path=None):
        # aux
        if cfg["task"]["aux_target"] is not None:
            cfg["task"]["aux_target_nclasses"] = []
            for t in cfg["task"]["aux_target"]:
                if t == "site_id":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "laterality":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "view":
                    cfg["task"]["aux_target_nclasses"].append(6)
                elif t == "implant":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "biopsy":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "invasive":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "BIRADS":
                    cfg["task"]["aux_target_nclasses"].append(4)
                elif t == "density":
                    cfg["task"]["aux_target_nclasses"].append(5)
                elif t == "difficult_negative_case":
                    cfg["task"]["aux_target_nclasses"].append(2)
                elif t == "age":
                    cfg["task"]["aux_target_nclasses"].append(10)
            print(cfg["task"]["aux_target_nclasses"])
        else:
            aux = False
            
        self.weight_path = weight_path
        self.cfg = cfg
        if cfg["model"]["weighted_loss"]:
            self.cfg["model"]["loss_weights"] = torch.tensor(0.0, dtype=torch.float32)
        else:
            self.cfg["model"]["loss_weights"] = None
        self.model = CNNModel(self.cfg)
        self.trainer = Trainer(**self.cfg["pl_params"])

    def predict(self, test_X):
        test_dataset = RSNADataset(self.cfg, test_X)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, **self.cfg["test_loader"],
        )
        preds = self.trainer.predict(
            self.model, dataloaders=test_dataloader, ckpt_path=self.weight_path
        )
        preds = torch.cat(preds, axis=0)
        preds = preds.cpu().detach().numpy()

        return preds
