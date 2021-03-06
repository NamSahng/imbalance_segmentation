import argparse
import logging
import os
import random

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from utils.augmentation import (
    get_postaug,
    get_preaug,
    get_preprocessing,
    get_training_augmentation,
    get_validation_augmentation,
)
from utils.dataset import VOC_ProbDataset, VOCDataset
from utils.losses import DiceFocal
from utils.metrics import Multiclass_IoU_Dice
from utils.scheduler import PolyLR
from utils.train import TrainEpoch, ValidEpoch
from utils.util import get_labedict


def get_logger(output_loc):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(os.path.join(output_loc, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed):
    # ref: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    if not seed:
        seed = 10
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # upsample_bilinear2d_backward_out_cuda
    # https://discuss.pytorch.org/t/deterministic-behavior-using-bilinear2d/131355/5
    # torch.use_deterministic_algorithms(True)


def get_vc_df(cur_train_df, col):
    cnt_vc = pd.DataFrame(cur_train_df[col].value_counts())
    cnt_vc = cnt_vc.reset_index()
    cnt_vc.columns = [col, "cnt"]
    cnt_vc["weight"] = 1 / np.log(cnt_vc.cnt)
    cnt_vc["weight"] = cnt_vc["weight"] / cnt_vc["weight"].sum()
    return cnt_vc


def get_params():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_rs", type=int, default=1, help="random state of train validation split"
    )
    parser.add_argument(
        "--split_fnum",
        type=int,
        default=0,
        help="fold number for each random state split",
    )
    parser.add_argument(
        "--imbalance",
        type=bool,
        default=False,
        help="whether make data imbalance intentionally or not",
    )
    # random, skf_prop, skf_major, mskf
    parser.add_argument(
        "--split",
        type=str,
        default="mskf",
        help="train validation splitting strategies",
    )
    # x2, cluster, cnt
    parser.add_argument(
        "--resampling_strategy",
        type=str,
        default="cnt",
        help="set resampling strategy for dataset_b",
    )
    parser.add_argument(
        "--copypaste", type=bool, default=True, help="use copy paste augmentation"
    )
    # cnt, performance
    parser.add_argument(
        "--paste_by",
        type=str,
        default="performance",
        help="paste strategy of copy paste augmentation",
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--encoder", type=str, default="resnet101")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--verbose", type=bool, default=True, help="")
    parser.add_argument("--copypaste_prop", type=float, default=1.0, help="")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="")
    parser.add_argument("--momentum", type=float, default=0.9, help="")
    parser.add_argument("--earlystopping_patience", type=int, default=11)
    parser.add_argument("--earlystopping_trigger", type=int, default=0)
    parser.add_argument("--earlystopping_eps", type=float, default=1e4)

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    # set seed
    seed = 2022
    seed_all(seed)

    # load data
    labeldict = get_labedict()
    label2num = labeldict["label2num"]
    num2label = labeldict["num2label"]
    label_names = label2num.keys()
    train_df = pd.read_csv("./data/train_df.csv", index_col=0)
    classes_of_interest = [
        label2num[k] for k in label2num.keys() if k not in ["border"]
    ]

    # get and set configs
    args = get_params()
    split_rs = args.split_rs
    split_fnum = args.split_fnum
    imbalance = args.imbalance
    split = args.split
    resampling_strategy = args.resampling_strategy
    copypaste = args.copypaste
    paste_by = args.paste_by
    imbalance = args.imbalance
    lr = args.lr  # 0.1
    encoder = args.encoder  # 'resnet101'
    encoder_weights = args.encoder_weights  # 'imagenet'
    device = args.device  # 'cuda'
    verbose = args.verbose  # False
    copypaste_prop = args.copypaste_prop  #  None if copypaste == 'FALSE' else 1.0
    weight_decay = args.weight_decay  # 1e-4
    momentum = args.momentum  # 0.9
    earlystopping_patience = args.earlystopping_patience  # 11
    earlystopping_trigger = args.earlystopping_trigger  # 0
    earlystopping_eps = args.earlystopping_eps  # 1e-4
    batch_size = 8
    num_epoch = 200

    # make output directory
    folder = "trial"
    output_dir = f"./output_single/{folder}"
    os.makedirs(output_dir, exist_ok=True)
    # get logger
    logger = get_logger(output_dir)

    # determine paste classes and its probability by cnt_df
    if paste_by == "cnt":
        cnt_df = pd.read_csv("./data/imbalance_df.csv", index_col=0)
        paste_dict = dict(
            cnt_df[cnt_df["after"] < 60]["after"].apply(lambda x: int(100 / x))
        )
        paste_list = []
        for label, num in paste_dict.items():
            paste_list += [label] * num
    # determine paste classes by segmentation performance
    elif paste_by == "performance":
        result_df = pd.read_csv("./data/previous_results.csv", index_col=0)
        paste_list = list(result_df.mean(axis=1).sort_values()[:8].index)
    else:
        paste_list = None

    # make data imbalance intentionally
    if imbalance:
        print(f"before imbalance {len(train_df)}")
        train_df = train_df[train_df["use_image"] == True]
        print(f"after imbalance {len(train_df)}")

    # get train and validation split by configs
    cur_split = f"{split}_rstate{split_rs}_fold{split_fnum}"
    cur_train_df = train_df[train_df[cur_split] == "train"]
    cur_val_df = train_df[train_df[cur_split] == "val"]

    # resampling stratey for dataset_b
    if resampling_strategy == "x2":
        vc_col = None
        vc_df = None
    elif resampling_strategy == "cluster":
        vc_col = "cluster_num_16"
        vc_df = get_vc_df(cur_train_df, vc_col)
    elif resampling_strategy == "cnt":
        vc_col = "major_label"
        vc_df = get_vc_df(cur_train_df, vc_col)

    # make dataset and dataloader by defined configs
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    dataset_a = VOC_ProbDataset(
        cur_train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        copypaste_prop=copypaste_prop,
        paste_list=paste_list,
        pre_aug=get_preaug,
        post_aug=get_postaug(),
        label2num=label2num,
    )
    dataset_b = VOC_ProbDataset(
        cur_train_df,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        copypaste_prop=copypaste_prop,
        paste_list=paste_list,
        pre_aug=get_preaug,
        post_aug=get_postaug(),
        label2num=label2num,
        vc_col=vc_col,
        vc_df=vc_df,
    )
    train_dataset = ConcatDataset([dataset_a, dataset_b])
    valid_dataset = VOCDataset(
        cur_val_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
    )
    # valid dataset's batch size should be 1 for class metrics
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

    # create segmentation model with pretrained encoder
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(label_names),
        activation="softmax2d",
    )

    # set metric and loss
    dice = smp.losses.DiceLoss(
        mode="multiclass", from_logits=False, ignore_index=label2num["border"]
    )
    focal = smp.losses.FocalLoss(mode="multiclass", ignore_index=label2num["border"])
    # loss = DiceFocal(dice, focal, len(classes_of_interest))
    loss = DiceFocal(dice, focal)
    loss_name, metric_name = "loss", "m_IoU"
    loss_names = [loss_name, "dice_loss", "focal_loss"]
    metrics = [
        Multiclass_IoU_Dice(
            mean_score=True,
            nan_score_on_empty=True,
            classes_of_interest=classes_of_interest,
            name=metric_name,
        )
    ]
    class_metrics = Multiclass_IoU_Dice(
        mean_score=False,
        nan_score_on_empty=True,
        classes_of_interest=classes_of_interest,
        name=metric_name,
        class_names= [num2label[i] for i in classes_of_interest] # label_names without borders
    )

    optimizer = torch.optim.SGD(
        params=[
            {"params": model.encoder.parameters(), "lr": 0.1 * lr},
            {"params": model.decoder.parameters(), "lr": lr},
            {"params": model.segmentation_head.parameters(), "lr": lr},
        ],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    train_epoch = TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=device,
        verbose=verbose,
        loss_names=loss_names,
        class_metrics=class_metrics
    )
    valid_epoch = ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=device,
        verbose=verbose,
        loss_names=loss_names,
        class_metrics=class_metrics
    )

    scheduler = PolyLR(optimizer, int(num_epoch * len(train_dataset) / batch_size))
    max_score = 0
    logger.info("Train Started")
    for i in range(1, num_epoch + 1):
        logger.info("\nEpoch: {}".format(i))
        train_logs = train_epoch.run(train_loader)
        logger.info(train_logs)
        logger.info(
            f"train loss:{train_logs[f'{loss_name}']:.4f}, train iou:{train_logs[f'{metric_name}']:.4f}"
        )
        valid_logs = valid_epoch.run(valid_loader)
        logger.info(valid_logs)
        logger.info(
            f"valid loss:{valid_logs[f'{loss_name}']:.4f}, valid iou:{valid_logs[f'{metric_name}']:.4f}"
        )

        # Early Stopping
        if valid_logs[f"{metric_name}"] < max_score + earlystopping_eps:
            earlystopping_trigger += 1
            if earlystopping_trigger >= earlystopping_patience:
                logger.info(f"Ealry stopping!")
                break
        else:
            earlystopping_trigger = 0
        # Save best validation model
        # if (max_score < valid_logs[f'{metric_name}']) and (i > 50):
        if max_score < valid_logs[f"{metric_name}"]:
            max_score = valid_logs[f"{metric_name}"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info("Model saved!")

        scheduler.step()

    # test with best model on valdiation dataset
    test_df = pd.read_csv("./data/test_df.csv", index_col=0)
    best_model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(label_names),
        activation="softmax2d",
    )
    best_model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    test_dataset = VOCDataset(
        test_df,
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    test_metric = Multiclass_IoU_Dice(
        mean_score=False,
        nan_score_on_empty=True,
        classes_of_interest=classes_of_interest,
        name=metric_name,
    )
    results = {}
    best_model.to(device)
    test_metric.to(device)
    best_model.eval()
    for x, y_true, img_id in tqdm(test_loader):
        x, y_true = x.to(device), y_true.to(device)
        with torch.no_grad():
            y_pred = best_model.forward(x)
        result = test_metric(y_pred, y_true)
        results[img_id[0]] = result[0]

    result_df = pd.DataFrame.from_dict(results, orient="index")
    result_df.columns = [num2label[num] for num in classes_of_interest]
    result_df.to_csv(os.path.join(output_dir, "test_result.csv"))
    class_result = result_df.apply(lambda x: np.nanmean(x))
    mean_iou = np.mean(class_result)
    logger.info(class_result)
    logger.info(f"mean IoU: {mean_iou}")
