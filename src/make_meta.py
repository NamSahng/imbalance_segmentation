import glob
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils.util import get_num2color

NUM_SPLIT = 5


def make_label_dict():
    """
    Make Label name to number dict and vice versa
    label number's are from https://bo-10000.tistory.com/38
    """
    #
    labels_num = """Aeroplane	1
Bicycle	2
Bird	3
Boat	4
Bottle	5
Bus	6
Car	7
Cat	8
Chair	9
Cow	10
Diningtable	11
Dog	12
Horse	13
Motorbike	14
Person	15
Pottedplant	16
Sheep	17
Sofa	18
Train	19
Tvmonitor	20"""
    labels_num = labels_num.lower().split("\n")
    labels_num = np.array([c.split("\t") for c in labels_num])
    label2num = dict(zip(labels_num[:, 0], labels_num[:, 1].astype(int)))
    label2num["background"] = 0
    label2num["border"] = 255
    num2label = dict(zip(label2num.values(), label2num.keys()))
    label_dict = {"label2num": label2num, "num2label": num2label}
    label_dict_loc = "./data/label_dict.pkl"

    with open(label_dict_loc, "wb") as f:
        pickle.dump(label_dict, f)

    return num2label, label2num


def calc_kmeans_silhouette(df, cols, wo_bg_border):
    if wo_bg_border:
        cols = cols[:-2]
    X = df[cols].values

    kmeans_exp_list = []
    for cur_num in range(2, 30):
        km = KMeans(
            n_clusters=cur_num,
            init="k-means++",
            n_init=50,
            max_iter=1000,
            random_state=2022,
        )
        km.fit(X)
        euc_silhouette = metrics.silhouette_score(X, km.labels_, metric="euclidean")
        kmeans_exp_list.append([cur_num, euc_silhouette, km])
        print(f"at K: {cur_num} Silhouette Coefficient : {euc_silhouette :0.3f}")

    kmeans_exp_list = np.array(kmeans_exp_list)
    cur_df = pd.DataFrame(
        zip(kmeans_exp_list[:, 0], kmeans_exp_list[:, 1]),
        columns=["n_clusters", "euclidean silhouette"],
    )
    ax = cur_df[["euclidean silhouette"]].set_index(cur_df.n_clusters).plot()
    print()
    return kmeans_exp_list, ax


def apply_cluster(
    df, X, num_cluster, kmeans_exp_list, labels, num2color, figloc, wo_bg_border
):
    if wo_bg_border:
        labels = labels[:-2]
    cur_km = kmeans_exp_list[num_cluster - 2][-1]
    df[f"cluster_num_{num_cluster}"] = cur_km.predict(X)
    temp_df = pd.DataFrame(df[f"cluster_num_{num_cluster}"].value_counts().sort_index())
    freqs = temp_df.values.squeeze()
    print(temp_df)
    cluster_center_df = pd.DataFrame(cur_km.cluster_centers_.T, index=labels)
    cluster_center_df = cluster_center_df.T
    color = [
        "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b in num2color.values()
    ]
    ax = cluster_center_df.plot(
        kind="bar", stacked=True, figsize=(12, 6), color=color, edgecolor="black"
    )
    x_coords = [p.get_x() for p in ax.patches[:num_cluster]]
    y_coord = 0.5

    for i in range(num_cluster):
        ax.annotate(str(freqs[i]), (x_coords[i] + 0.1, y_coord))
    ax.legend(title="labels", bbox_to_anchor=(1.00, 1), loc="upper left")
    plt.savefig(figloc)
    return df


if __name__ == "__main__":

    # make meta file of images
    train_list = open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt")
    test_list = open("./data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt")
    train_list = train_list.readlines()
    test_list = test_list.readlines()
    train_list = [re.sub("\n", "", f) for f in train_list]
    test_list = [re.sub("\n", "", f) for f in test_list]
    train_df = pd.DataFrame(train_list, columns=["id"])
    test_df = pd.DataFrame(test_list, columns=["id"])
    train_df["ftype"] = "train"
    test_df["ftype"] = "test"
    data_df = pd.concat([train_df, test_df])

    image_dir = "./data/VOCdevkit/VOC2012/JPEGImages/"
    mask_dir = "./data/VOCdevkit/VOC2012/SegmentationClass/"
    images = glob.glob(image_dir + "*")
    masks = glob.glob(mask_dir + "*")

    # Check File Extension
    image_ext = list(set([m.split(".")[-1] for m in images]))
    mask_ext = list(set([m.split(".")[-1] for m in masks]))
    assert (len(image_ext) == 1) & (image_ext[0] == "jpg")
    assert (len(mask_ext) == 1) & (mask_ext[0] == "png")

    # make image location column
    data_df["image_loc"] = image_dir + data_df["id"] + ".jpg"
    data_df["mask_loc"] = mask_dir + data_df["id"] + ".png"

    # get num2label & label2num
    num2label, label2num = make_label_dict()

    # Make class information on image
    info_dict = {}
    for i in tqdm(range(len(data_df))):
        cur_id = data_df.iloc[i]["id"]
        img_loc = data_df.iloc[i]["image_loc"]
        mask_loc = data_df.iloc[i]["mask_loc"]

        mask = np.array(Image.open(mask_loc))
        mask_nums = np.unique(mask)

        h, w = mask.shape
        cur_meta = {}

        cur_meta["height"] = h
        cur_meta["width"] = w
        cur_meta["area"] = h * w

        for k in label2num.keys():
            cur_meta[f"{k}_sum"] = 0

        for num in mask_nums:
            pix_sum = np.sum(mask == num)
            cur_meta[f"{num2label[num]}_sum"] = pix_sum

        info_dict[cur_id] = cur_meta

    # Merge with meta_df
    meta_df = pd.DataFrame(info_dict).T.reset_index()
    meta_df = meta_df.rename(columns={"index": "id"})
    meta_df = pd.merge(data_df, meta_df, on="id")

    labels = label2num.keys()
    label_sum_cols = [f"{label}_sum" for label in labels]

    # make major_label column
    consider_sum_cols = label_sum_cols[:-2]
    meta_df["major_label"] = meta_df[consider_sum_cols].idxmax(axis=1)
    meta_df["major_label"] = meta_df["major_label"].apply(
        lambda x: x.replace("_sum", "")
    )
    meta_df["major_num"] = meta_df["major_label"].apply(lambda x: label2num[x])

    # make class proportion columns of its image
    for label in label_sum_cols:
        label_name = label.split("_")[0]
        meta_df[f"{label_name}_prop"] = meta_df[label] / meta_df["area"]

    # make frequency columns
    for label in label_sum_cols:
        label_name = label.split("_")[0]
        meta_df[f"{label_name}_cnt"] = meta_df[label] > 0

    label_prop_cols = [i for i in meta_df.columns if i.endswith("_prop")]
    label_cnt_cols = [i for i in meta_df.columns if i.endswith("_cnt")]

    # make use_image column to make data imbalance
    meta_df["use_image"] = True
    reduce_classes = "bird, cat, aeroplane, bicycle, bottle, chair".replace(
        ",", " "
    ).split()
    reduce_classes = [f"{i}_cnt" for i in reduce_classes]
    remain_prop = 1 / 5
    remove_prop = 1 - remain_prop
    np.random.seed(2022)
    temp = []
    for col in reduce_classes:
        meta_df[meta_df.ftype == "train"]
        train_col_id = meta_df[
            (meta_df.ftype == "train")
            & (meta_df[col] == True)
            & (meta_df["use_image"] == True)
        ].id
        train_num = len(train_col_id)
        remove_id = np.random.choice(
            list(train_col_id), int(train_num * remove_prop), replace=False
        )
        meta_df.loc[meta_df.id.isin(remove_id), "use_image"] = False
        temp.append(remove_id)

    # check frequency of labels on data
    # 7.pg https://pjreddie.com/media/files/VOC2012_doc.pdf
    train_df = meta_df[(meta_df["ftype"] == "train")]
    cnt_1 = train_df[label_cnt_cols].sum()
    train_use_df = meta_df[(meta_df["ftype"] == "train") & meta_df["use_image"]]
    cnt_2 = train_use_df[label_cnt_cols].sum()
    cnt_df = pd.concat([cnt_1, cnt_2], axis=1)
    cnt_df.columns = ["before", "after"]
    cnt_df.index = [i.replace("_cnt", "") for i in cnt_df.index]
    print(cnt_df)
    cnt_df.to_csv("./data/imbalance_df.csv")

    #  K-means cluster on image's label proportion and label frequency
    exps = []
    exps.append(calc_kmeans_silhouette(train_df, label_prop_cols, True))
    exps.append(calc_kmeans_silhouette(train_df, label_prop_cols, False))
    exps.append(calc_kmeans_silhouette(train_df, label_cnt_cols, True))
    exps.append(calc_kmeans_silhouette(train_df, label_cnt_cols, False))

    num2color = get_num2color(num2label)

    # apply cluster on images and check proportion of clusters
    num_cluster = 16
    X = meta_df[label_prop_cols[:-2]].values
    meta_df = apply_cluster(
        meta_df,
        X,
        num_cluster,
        exps[0][0],
        list(labels),
        num2color,
        f"./data/trainval_cluster_{num_cluster}.png",
        True,
    )

    train_df = meta_df[(meta_df["ftype"] == "train") & (meta_df["use_image"] == True)]
    X = train_df[label_prop_cols[:-2]].values
    train_df = apply_cluster(
        train_df,
        X,
        num_cluster,
        exps[0][0],
        list(labels),
        num2color,
        f"./data/train_cluster_{num_cluster}.png",
        True,
    )

    # split data randomly, stratified
    train_df = meta_df[meta_df.ftype == "train"].copy()
    test_df = meta_df[meta_df.ftype == "test"].copy()
    for rs in range(1, 6):
        kfold = KFold(n_splits=NUM_SPLIT, random_state=rs, shuffle=True)
        skf = StratifiedKFold(n_splits=NUM_SPLIT, random_state=rs, shuffle=True)
        mskf = MultilabelStratifiedKFold(
            n_splits=NUM_SPLIT, random_state=rs, shuffle=True
        )
        label_onehot_values = train_df[label_cnt_cols].astype(int).values

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_df.index)):
            train_df.loc[
                train_df.index.isin(train_idx), f"random_rstate{rs}_fold{fold}"
            ] = "train"
            train_df.loc[
                train_df.index.isin(val_idx), f"random_rstate{rs}_fold{fold}"
            ] = "val"
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df.index, train_df.cluster_num_16)
        ):
            train_df.loc[
                train_df.index.isin(train_idx), f"skf_prop_rstate{rs}_fold{fold}"
            ] = "train"
            train_df.loc[
                train_df.index.isin(val_idx), f"skf_prop_rstate{rs}_fold{fold}"
            ] = "val"
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(train_df.index, train_df.major_label)
        ):
            train_df.loc[
                train_df.index.isin(train_idx), f"skf_major_rstate{rs}_fold{fold}"
            ] = "train"
            train_df.loc[
                train_df.index.isin(val_idx), f"skf_major_rstate{rs}_fold{fold}"
            ] = "val"
        for fold, (train_idx, val_idx) in enumerate(
            mskf.split(train_df.index, label_onehot_values)
        ):
            train_df.loc[
                train_df.index.isin(train_idx), f"mskf_rstate{rs}_fold{fold}"
            ] = "train"
            train_df.loc[
                train_df.index.isin(val_idx), f"mskf_rstate{rs}_fold{fold}"
            ] = "val"

    train_df.to_csv("./data/train_df.csv")
    test_df.to_csv("./data/test_df.csv")
