import argparse
import os

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from classifier import CNNClassifier


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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config (.yaml)")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument("-t", "--tta", action="store_true")
    parser.add_argument("-r", "--resume_train", action="store_true")
    args = parser.parse_args()
    return args


def wandb_start(cfg):
    wandb.init(
        project=cfg["general"]["project_name"],
        name=f"{cfg['general']['save_name']}_{cfg['fold_n']}",
        group=f"{cfg['general']['save_name']}_cv" if cfg["general"]["cv"] else "all",
        job_type=cfg["job_type"],
        mode="disabled" if cfg["general"]["wandb_desabled"] else "online",
        config=cfg,
    )


def train_and_predict(cfg, train_X, train_y, valid_X=None, valid_y=None):
    model = CNNClassifier(cfg, train_X, train_y, valid_X=valid_X, valid_y=valid_y)
    model.train(weight_path=cfg["ckpt_path"])

    if valid_X is None:
        del model
        torch.cuda.empty_cache()
        return
    else:
        valid_preds = model.predict(valid_X)
        del model
        torch.cuda.empty_cache()
        return valid_preds


def one_fold(sgkf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    train_indices, valid_indices = list(
        sgkf.split(train_X, train_y, groups=train_X["patient_id"]) #train_X["stratify"].values
    )[fold_n]
    train_X_cv, train_y_cv = (
        train_X.iloc[train_indices].reset_index(drop=True),
        train_y.iloc[train_indices].reset_index(drop=True),
    )
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )
    
    # retrieve anomaly data
    if not train_X_cv[train_X_cv["patient_id"]==27770].empty:
        drop_idx = train_X_cv[train_X_cv["patient_id"]==27770].index
        train_X_cv = train_X_cv.drop(train_X_cv.index[drop_idx]).reset_index(drop=True)
        train_y_cv = train_y_cv.drop(train_y_cv.index[drop_idx]).reset_index(drop=True)
    else:
        drop_idx = valid_X_cv[valid_X_cv["patient_id"]==27770].index
        valid_X_cv = valid_X_cv.drop(valid_X_cv.index[drop_idx]).reset_index(drop=True)
        valid_y_cv = valid_y_cv.drop(valid_y_cv.index[drop_idx]).reset_index(drop=True)
    if not train_X_cv[train_X_cv["image_id"]==1942326353].empty:
        drop_idx = train_X_cv[train_X_cv["image_id"]==1942326353].index
        train_X_cv = train_X_cv.drop(train_X_cv.index[drop_idx]).reset_index(drop=True)
        train_y_cv = train_y_cv.drop(train_y_cv.index[drop_idx]).reset_index(drop=True)
    else:
        drop_idx = valid_X_cv[valid_X_cv["image_id"]==1942326353].index
        valid_X_cv = valid_X_cv.drop(valid_X_cv.index[drop_idx]).reset_index(drop=True)
        valid_y_cv = valid_y_cv.drop(valid_y_cv.index[drop_idx]).reset_index(drop=True)

    # check
    print(f"train_len{len(train_y_cv)}")
    print(train_y_cv.value_counts())
    print(train_X_cv["site_id"].value_counts())
    print(f"valid_len{len(valid_y_cv)}")
    print(valid_y_cv.value_counts())
    print(valid_X_cv["site_id"].value_counts())
    valid_preds = train_and_predict(cfg, train_X_cv, train_y_cv, valid_X_cv, valid_y_cv)

    # print
    print(f"[fold_{fold_n}]")
    pfbeta_score = pfbeta(valid_y_cv.values.flatten(), valid_preds.flatten())
    optimized_pfbeta_score, threshold = optimal_f1(valid_y_cv.values.flatten(), valid_preds.flatten())
    auc_score = roc_auc_score(valid_y_cv.values.flatten(), valid_preds.flatten())
    print(f"[per image], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
    wandb.log({"pf1_score_per_image": pfbeta_score, "auc_score_per_image": auc_score})
    wandb.log({"pf1_score_per_image_optimized": optimized_pfbeta_score, "threshold_per_image": threshold})
    # aggregation max
    valid_X_cv["true_target"] = valid_y_cv
    valid_X_cv["preds_target"] = valid_preds
    valid_X_cv_1 = valid_X_cv[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).max().reset_index()
    pfbeta_score = pfbeta(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    optimized_pfbeta_score, threshold = optimal_f1(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    auc_score = roc_auc_score(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    print(f"[per patient laterality max], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
    wandb.log({"pf1_score_per_patient_max": pfbeta_score, "auc_score_per_patient_max": auc_score})
    wandb.log({"pf1_score_per_patient_max_optimized": optimized_pfbeta_score, "threshold_per_patient_max": threshold})
    # aggregation mean
    valid_X_cv_2 = valid_X_cv[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).mean().reset_index()
    pfbeta_score = pfbeta(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    optimized_pfbeta_score, threshold = optimal_f1(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    auc_score = roc_auc_score(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    print(f"[per patient laterality mean], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
    wandb.log({"pf1_score_per_patient_mean": pfbeta_score, "auc_score_per_patient_mean": auc_score})
    wandb.log({"pf1_score_per_patient_mean_optimized": optimized_pfbeta_score, "threshold_per_patient_mean": threshold})

    torch.cuda.empty_cache()
    wandb.finish()

    return valid_preds, pfbeta_score, auc_score


def all_train(cfg, train_X, train_y):
    print("[all_train] start")
    seed_everything(cfg["general"]["seed"], workers=True)

    # retrieve anomaly data
    if not train_X[train_X["patient_id"]==27770].empty:
        drop_idx = train_X[train_X["patient_id"]==27770].index
        train_X = train_X.drop(train_X.index[drop_idx]).reset_index(drop=True)
        train_y = train_y.drop(train_y.index[drop_idx]).reset_index(drop=True)
    if not train_X[train_X["image_id"]==1942326353].empty:
        drop_idx = train_X[train_X["image_id"]==1942326353].index
        train_X = train_X.drop(train_X.index[drop_idx]).reset_index(drop=True)
        train_y = train_y.drop(train_y.index[drop_idx]).reset_index(drop=True)

    # check
    print(f"train_len{len(train_y)}")
    print(train_y.value_counts())
    print(train_X["site_id"].value_counts())
    
    # train
    train_and_predict(cfg, train_X, train_y)

    return


def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")
    cfg["tta"] = args.tta
    if args.tta:
        print("using tta")

    # set jobtype for wandb
    cfg["job_type"] = "train"

    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train = pd.read_csv(f"{cfg['general']['input_path']}/train.csv")
    
    # for auc targets
    if cfg["task"]["aux_target"] is not None:
        train.age.fillna(train.age.mean(), inplace=True)
        train["age"] = pd.qcut(train.age, 10, labels=range(10), retbins=False).astype(int)
        train[cfg["task"]["aux_target"]] = train[cfg["task"]["aux_target"]].apply(LabelEncoder().fit_transform)

    # split X/y
    #train["stratify"] = train[cfg["task"]["target"]].astype(str) + train["site_id"].astype(str)
    """
    num_bins = 5
    train["age_bin"] = pd.cut(train["age"].values.reshape(-1), bins=num_bins, labels=False)
    strat_cols = [
        "laterality", "view", "biopsy", "invasive", "BIRADS", "age_bin",
        "implant", "density","machine_id", "difficult_negative_case",
        "cancer",
    ]
    train['stratify'] = ""
    for col in strat_cols:
        train["stratify"] += train[col].astype(str)
    """
    print(train["age"].value_counts())
    
    train_X = train.drop(cfg["task"]["target"], axis=1)
    train_y = train[cfg["task"]["target"]]

    if cfg["general"]["cv"]:        
        sgkf = StratifiedGroupKFold(
            n_splits=cfg["general"]["n_splits"],
            shuffle=True,
            random_state=cfg["general"]["seed"],
        )
        valid_pfbeta_list = []
        valid_auc_list = []
        for fold_n in tqdm(cfg["general"]["fold"]):
            cfg["fold_n"] = fold_n
            
            if args.resume_train and os.path.isfile(f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}.ckpt"):
                print("resume train")
                cfg["ckpt_path"] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}.ckpt"
            else:
                cfg["ckpt_path"] = None
            
            _, pfbeta_score, auc_score = one_fold(sgkf, cfg, train_X, train_y, fold_n)
            valid_pfbeta_list.append(pfbeta_score)
            valid_auc_list.append(auc_score)

        valid_pfbeta_mean = np.mean(valid_pfbeta_list, axis=0)
        valid_auc_mean = np.mean(valid_auc_list, axis=0)
        print(f"cv mean pfbeta:{valid_pfbeta_mean}")
        print(f"cv mean auc:{valid_auc_mean}")
    else:
        # train all data
        cfg["fold_n"] = "all"
        
        if args.resume_train and os.path.isfile(f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch.ckpt"):
            print("resume train")
            cfg["ckpt_path"] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch.ckpt"
        else:
            cfg["ckpt_path"] = None
        
        all_train(cfg, train_X, train_y)


if __name__ == "__main__":
    main()
