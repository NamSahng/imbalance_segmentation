import argparse
import logging
import os
import random

import nni
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from nni.utils import merge_parameter
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

logger = logging.getLogger("imbalace_seg_NNI")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_all(seed):
    # ref: https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
    if not seed:
        seed = 10
    print("[ Using Seed : ", seed, " ]")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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


# def get_params():
#     # Training settings
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--split_rs", type=int, default=1, help="random state of train validation split"
#     )
#     parser.add_argument(
#         "--split_fnum",
#         type=int,
#         default=0,
#         help="fold number for each random state split",
#     )
#     parser.add_argument(
#         "--configs",
#         type=str,
#         default="FALSE random x2 FALSE none",
#         help="(imbalance split strategies resampling_strategy copypaste paste_by)",
#     )
#     args, _ = parser.parse_known_args()
#     return args
def get_params():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float,
    )
    parser.add_argument(
        "--loss", type=str,
    )
    parser.add_argument(
        "--encoder_lr", type=float,
    )
    args, _ = parser.parse_known_args()
    return args

def main(args):
    # load data
    labeldict = get_labedict()
    label2num = labeldict["label2num"]
    num2label = labeldict["num2label"]
    label_names = label2num.keys()
    train_df = pd.read_csv("./data/train_df.csv", index_col=0)
    classes_of_interest = [
        label2num[k] for k in label2num.keys() if k not in ["border"]
    ]

    split_rs = 1
    split_fnum = 0
    configs = "FALSE mskf x2 FALSE none"
    imbalance, split, resampling_strategy, copypaste, paste_by = configs.split(" ")
    imbalance = True if imbalance == "TRUE" else False
    lr = args['lr'] # 0.1 0.01 -> model couldn't predict on minor classes on test data
    encoder = "resnet101"
    encoder_weights =  'imagenet'
    device = "cuda"
    verbose = False
    copypaste_prop = None if copypaste == "FALSE" else 1.0
    weight_decay = 1e-4
    momentum = 0.9
    earlystopping_patience = 11
    earlystopping_trigger = 0
    earlystopping_eps = 1e-3
    batch_size = 4
    num_epoch = 3

    # make output directory
    # folder = args["configs"].replace(" ", "_")
    from datetime import datetime
    cur_time = datetime.now()
    folder = cur_time.strftime("%y%m%d_%H%M%S")
    folder = f"{args['lr']}_{args['loss']}_{args['encoder_lr']}".replace('.', '_')
    output_dir = f"./output/{folder}"
    os.makedirs(output_dir, exist_ok=True)

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
        # activation="softmax2d",
    )

    # set metric and loss
    if args['loss'] == 'dicefocal':
        dice = smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=label2num["border"]
        )
        focal = smp.losses.FocalLoss(mode="multiclass", ignore_index=label2num["border"])
        loss = DiceFocal(dice, focal)
        loss_name, metric_name = "loss", "m_IoU"
        loss_names = [loss_name, "dice_loss", "focal_loss"]
    elif args['loss'] == 'dice':
        loss =  smp.losses.DiceLoss(
            mode="multiclass", from_logits=True, ignore_index=label2num["border"]
        )
        loss.__name__ = "loss"
        loss_name, metric_name = "loss", "m_IoU"
        loss_names = None
    elif args['loss'] == 'focal':
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=label2num["border"])
        loss.__name__ = "loss"
        loss_name, metric_name = "loss", "m_IoU"
        loss_names = None
    
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
            {"params": model.encoder.parameters(), "lr": args['encoder_lr']},
            # {"params": model.encoder.parameters(), "lr": lr},
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

    # scheduler = PolyLR(optimizer, int(num_epoch * len(train_dataset) / batch_size))
    scheduler = PolyLR(optimizer, int(200 * len(train_dataset) / batch_size))

    max_score = 0
    logger.info("Train Started")
    for i in range(1, num_epoch + 1):
        logger.info("="*64)
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
        nni.report_intermediate_result(valid_logs[f"{metric_name}"])

        # Early Stopping
        if valid_logs[f"{metric_name}"] < max_score + earlystopping_eps:
            earlystopping_trigger += 1
            if earlystopping_trigger >= earlystopping_patience:
                nni.report_intermediate_result(max_score)
                logger.info(f"Ealry stopping!")
                break
        else:
            earlystopping_trigger = 0
        # Save best validation model
        #if (max_score < valid_logs[f"{metric_name}"]) and (i > 50):
        if (max_score < valid_logs[f"{metric_name}"]):
            max_score = valid_logs[f"{metric_name}"]
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            logger.info("Model saved!")
        cur_encoder_lr = scheduler.optimizer.param_groups[0]['lr']
        cur_decoder_lr = scheduler.optimizer.param_groups[1]['lr']
        logger.info(f"encoder lr: {cur_encoder_lr:.4f}, decoder lr: {cur_decoder_lr:.4f}")
        logger.info("="*64)
        scheduler.step()

    # test with best model on valdiation dataset
    test_df = pd.read_csv("./data/test_df.csv", index_col=0)
    best_model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(label_names),
        #activation="softmax2d",
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
    nni.report_final_result(mean_iou)


if __name__ == "__main__":
    # set seed
    seed = 2022
    seed_all(seed)
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
