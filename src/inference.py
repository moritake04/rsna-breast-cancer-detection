import argparse
import gc

import joblib
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
import sklearn.metrics
from pytorch_lightning import seed_everything
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from classifier import CNNClassifierInference


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
    parser.add_argument("mode", type=str, help="valid or test or both")
    parser.add_argument("-f", "--fold", type=int, help="fold")
    parser.add_argument("-s", "--save_preds", action="store_true", help="Whether to save the predicted value or not.")
    parser.add_argument("-a", "--amp", action="store_false")
    #parser.add_argument("-t", "--tta", action="store_true")
    parser.add_argument("-t", "--tta", type=str, help="choose tta method")
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
    
def one_fold_valid(sgkf, cfg, train_X, train_y, fold_n):
    print(f"[fold_{fold_n}] start")
    seed_everything(cfg["general"]["seed"], workers=True)
    _, valid_indices = list(
        sgkf.split(train_X, train_y, groups=train_X["patient_id"])
    )[fold_n]
    valid_X_cv, valid_y_cv = (
        train_X.iloc[valid_indices].reset_index(drop=True),
        train_y.iloc[valid_indices].reset_index(drop=True),
    )
    
    # retrieve anomaly data
    if not valid_X_cv[valid_X_cv["patient_id"]==27770].empty:
        drop_idx = valid_X_cv[valid_X_cv["patient_id"]==27770].index
        print(drop_idx)
        valid_X_cv = valid_X_cv.drop(valid_X_cv.index[drop_idx]).reset_index(drop=True)
        valid_y_cv = valid_y_cv.drop(valid_y_cv.index[drop_idx]).reset_index(drop=True)
    if not valid_X_cv[valid_X_cv["image_id"]==1942326353].empty:
        drop_idx = valid_X_cv[valid_X_cv["image_id"]==1942326353].index
        print(drop_idx)
        valid_X_cv = valid_X_cv.drop(valid_X_cv.index[drop_idx]).reset_index(drop=True)
        valid_y_cv = valid_y_cv.drop(valid_y_cv.index[drop_idx]).reset_index(drop=True)

    model = CNNClassifierInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    valid_preds = model.predict(valid_X_cv)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    if cfg["save_preds"]:
        print("save_preds!")
        joblib.dump(valid_preds, f"{cfg['general']['output_path']}/preds/valid_{cfg['general']['seed']}_{cfg['general']['save_name']}_{fold_n}.preds", compress=3)

    print(f"[fold_{fold_n}]")
    pfbeta_score = pfbeta(valid_y_cv.values.flatten(), valid_preds.flatten())
    optimized_pfbeta_score, threshold_1 = optimal_f1(valid_y_cv.values.flatten(), valid_preds.flatten())
    auc_score = roc_auc_score(valid_y_cv.values.flatten(), valid_preds.flatten())
    recall = sklearn.metrics.recall_score(valid_y_cv.values.flatten(), valid_preds.flatten() > threshold_1)
    specificity = sklearn.metrics.recall_score(valid_y_cv.values.flatten(), valid_preds.flatten() > threshold_1, pos_label=0)
    precision = sklearn.metrics.precision_score(valid_y_cv.values.flatten(), valid_preds.flatten() > threshold_1)
    print(f"[per image], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold_1}")
    print(f"optimize recall {recall}")
    print(f"optimize specificity {specificity}")
    print(f"optimize precision {precision}")
    # aggregation max
    valid_X_cv["true_target"] = valid_y_cv
    valid_X_cv["preds_target"] = valid_preds
    valid_X_cv_1 = valid_X_cv[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).max().reset_index()
    pfbeta_score = pfbeta(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    optimized_pfbeta_score, threshold_2 = optimal_f1(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    auc_score = roc_auc_score(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten())
    recall = sklearn.metrics.recall_score(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten() > threshold_2)
    specificity = sklearn.metrics.recall_score(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten() > threshold_2, pos_label=0)
    precision = sklearn.metrics.precision_score(valid_X_cv_1["true_target"].values.flatten(), valid_X_cv_1["preds_target"].values.flatten() > threshold_2)
    print(f"[per patient laterality max], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold_2}")
    print(f"optimize recall {recall}")
    print(f"optimize specificity {specificity}")
    print(f"optimize precision {precision}")
    # aggregation mean
    valid_X_cv_2 = valid_X_cv[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).mean().reset_index()
    pfbeta_score = pfbeta(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    optimized_pfbeta_score, threshold_3 = optimal_f1(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    auc_score = roc_auc_score(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten())
    recall = sklearn.metrics.recall_score(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten() > threshold_3)
    specificity = sklearn.metrics.recall_score(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten() > threshold_3, pos_label=0)
    precision = sklearn.metrics.precision_score(valid_X_cv_2["true_target"].values.flatten(), valid_X_cv_2["preds_target"].values.flatten() > threshold_3)
    print(f"[per patient laterality mean], pfbeta:{pfbeta_score}, AUC:{auc_score}")
    print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold_3}")
    print(f"optimize recall {recall}")
    print(f"optimize specificity {specificity}")
    print(f"optimize precision {precision}")

    return valid_X_cv, threshold_1, threshold_2, threshold_3

def one_fold_test(cfg, test_X, fold_n):
    print(f"[fold_{fold_n}]")
    seed_everything(cfg["general"]["seed"], workers=True)

    model = CNNClassifierInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(test_X)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return test_preds

def all_train_test(cfg, test_X):
    print("[all_train]")

    seed_everything(cfg["general"]["seed"], workers=True)

    model = CNNClassifierInference(cfg, f"{cfg['ckpt_path']}.ckpt")
    test_preds = model.predict(test_X)

    return test_preds

def main():
    # read config
    args = get_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.fold is not None:
        cfg["general"]["fold"] = [args.fold]
    print(f"fold: {cfg['general']['fold']}")
    cfg["mode"] = args.mode
    cfg["save_preds"] = args.save_preds
    
    if args.tta == "flip" or args.tta == "clahe":
        print(f"using tta: {args.tta}")
        cfg["tta"] = args.tta
    else:
        cfg["tta"] = False
    
    if args.amp:
        print("fp16")
        cfg["pl_params"]["precision"] = 16
    else:
        print("fp32")
        cfg["pl_params"]["precision"] = 32
        
    # random seed setting
    seed_everything(cfg["general"]["seed"], workers=True)

    # read csv
    train = pd.read_csv(f"{cfg['general']['input_path']}/train.csv")
    test_X = pd.read_csv(f"{cfg['general']['input_path']}/test.csv")

    # split X/y
    train_X = train.drop(cfg["task"]["target"], axis=1)
    train_y = train[cfg["task"]["target"]]
    
    if cfg["general"]["cv"]:
        if cfg["mode"] == "valid":
            sgkf = StratifiedGroupKFold(
                n_splits=cfg["general"]["n_splits"],
                shuffle=True,
                random_state=cfg["general"]["seed"],
            )
            valid_X = pd.DataFrame(index=[], columns=train_X.columns)
            valid_X_list = []
            thr_list_1 = []
            thr_list_2 = []
            thr_list_3 = []
            #valid_pfbeta_list = []
            #valid_auc_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg["ckpt_path"] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                valid_X_cv, thr_1, thr_2, thr_3 = one_fold_valid(sgkf, cfg, train_X, train_y, fold_n)
                valid_X = pd.concat([valid_X, valid_X_cv])
                valid_X_list.append(valid_X_cv)
                thr_list_1.append(thr_1)
                thr_list_2.append(thr_2)
                thr_list_3.append(thr_3)
                #valid_pfbeta_list.append(pfbeta_score)
                #valid_auc_list.append(auc_score)

            thr_mean_1 = np.mean(thr_list_1, axis=0)
            thr_mean_2 = np.mean(thr_list_2, axis=0)
            thr_mean_3 = np.mean(thr_list_3, axis=0)
            print(f"cv mean thr per image:{thr_mean_1}")
            print(f"cv mean thr patient laterality max:{thr_mean_2}")
            print(f"cv mean thr patient laterality mean:{thr_mean_3}")
            #valid_pfbeta_mean = np.mean(valid_pfbeta_list, axis=0)
            #valid_auc_mean = np.mean(valid_auc_list, axis=0)
            #print(f"cv mean pfbeta:{valid_pfbeta_mean}")
            #print(f"cv mean auc:{valid_auc_mean}")
            print()
            print(f"↓ cv mean using optimized threshold")
            valid_pfbeta_list = []
            valid_pfbeta_list_max = []
            valid_pfbeta_list_mean = []
            valid_auc_list = []
            valid_auc_list_max = []
            valid_auc_list_mean = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                # normal
                pfbeta_score = pfbeta(valid_X_list[fold_n]["true_target"].values.flatten(), valid_X_list[fold_n]["preds_target"].values.flatten() > thr_mean_1)
                auc_score = roc_auc_score(valid_X_list[fold_n]["true_target"].values.flatten(), valid_X_list[fold_n]["preds_target"].values.flatten())
                valid_pfbeta_list.append(pfbeta_score)
                valid_auc_list.append(auc_score)
                # aggregation max
                valid_X_1 = valid_X_list[fold_n][["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).max().reset_index()
                pfbeta_score = pfbeta(valid_X_1["true_target"].values.flatten(), valid_X_1["preds_target"].values.flatten() > thr_mean_2)
                auc_score = roc_auc_score(valid_X_1["true_target"].values.flatten(), valid_X_1["preds_target"].values.flatten())
                valid_pfbeta_list_max.append(pfbeta_score)
                valid_auc_list_max.append(auc_score)
                # aggregation mean
                valid_X_2 = valid_X_list[fold_n][["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).mean().reset_index()
                pfbeta_score = pfbeta(valid_X_2["true_target"].values.flatten(), valid_X_2["preds_target"].values.flatten() > thr_mean_3)
                auc_score = roc_auc_score(valid_X_2["true_target"].values.flatten(), valid_X_2["preds_target"].values.flatten())
                valid_pfbeta_list_mean.append(pfbeta_score)
                valid_auc_list_mean.append(auc_score)
            valid_pfbeta_mean = np.mean(valid_pfbeta_list, axis=0)
            valid_pfbeta_mean_max = np.mean(valid_pfbeta_list_max, axis=0)
            valid_pfbeta_mean_mean = np.mean(valid_pfbeta_list_mean, axis=0)
            valid_auc_mean = np.mean(valid_auc_list, axis=0)
            valid_auc_mean_max = np.mean(valid_auc_list_max, axis=0)
            valid_auc_mean_mean = np.mean(valid_auc_list_mean, axis=0)
            print(f"[per image], cv mean pfbeta:{valid_pfbeta_mean}")
            print(f"[per patient laterality max], cv mean pfbeta:{valid_pfbeta_mean_max}")
            print(f"[per patient laterality mean], cv mean pfbeta:{valid_pfbeta_mean_mean}")
            print(f"[per image], cv mean auc:{valid_auc_mean}")
            print(f"[per patient laterality max], cv mean auc:{valid_auc_mean_max}")
            print(f"[per patient laterality mean], cv mean auc:{valid_auc_mean_mean}")
            print()
            print(f"↓ compute threshold after concatenate")
            # normal
            pfbeta_score = pfbeta(valid_X["true_target"].values.flatten(), valid_X["preds_target"].values.flatten())
            optimized_pfbeta_score, threshold = optimal_f1(valid_X["true_target"].values.flatten(), valid_X["preds_target"].values.flatten())
            auc_score = roc_auc_score(valid_X["true_target"].values.flatten(), valid_X["preds_target"].values.flatten())
            print(f"[per image], pfbeta:{pfbeta_score}, AUC:{auc_score}")
            print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
            # aggregation max
            valid_X_1 = valid_X[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).max().reset_index()
            pfbeta_score = pfbeta(valid_X_1["true_target"].values.flatten(), valid_X_1["preds_target"].values.flatten())
            optimized_pfbeta_score, threshold = optimal_f1(valid_X_1["true_target"].values.flatten(), valid_X_1["preds_target"].values.flatten())
            auc_score = roc_auc_score(valid_X_1["true_target"].values.flatten(), valid_X_1["preds_target"].values.flatten())
            print(f"[per patient laterality max], pfbeta:{pfbeta_score}, AUC:{auc_score}")
            print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
            # aggregation mean
            valid_X_2 = valid_X[["patient_id", "laterality", "true_target", "preds_target"]].groupby(["patient_id", "laterality"]).mean().reset_index()
            pfbeta_score = pfbeta(valid_X_2["true_target"].values.flatten(), valid_X_2["preds_target"].values.flatten())
            optimized_pfbeta_score, threshold = optimal_f1(valid_X_2["true_target"].values.flatten(), valid_X_2["preds_target"].values.flatten())
            auc_score = roc_auc_score(valid_X_2["true_target"].values.flatten(), valid_X_2["preds_target"].values.flatten())
            print(f"[per patient laterality mean], pfbeta:{pfbeta_score}, AUC:{auc_score}")
            print(f"optimize: {optimized_pfbeta_score}, threshold:{threshold}")
        
        elif cfg["mode"] == "test":
            test_preds_list = []
            for fold_n in tqdm(cfg["general"]["fold"]):
                cfg["fold_n"] = fold_n
                cfg["ckpt_path"] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch_fold{fold_n}"
                test_preds = one_fold_test(cfg, test_X, fold_n)
                test_preds_list.append(test_preds)
                print(test_preds)

            final_test_preds = np.mean(test_preds_list, axis=0)
            print(final_test_preds)
    else:
        # train all data
        cfg["fold_n"] = "all"
        cfg["ckpt_path"] = f"{cfg['general']['output_path']}/weights/{cfg['general']['save_name']}/last_epoch"
        final_test_preds = all_train_test(cfg, test_X)
        print(final_test_preds)

if __name__ == "__main__":
    main()