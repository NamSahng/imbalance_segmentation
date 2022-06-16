import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


def get_labedict():
    label_dict_loc = "./data/label_dict.pkl"
    with open(label_dict_loc, "rb") as f:
        labeldict = pickle.load(f)
    return labeldict


def get_num2color(num2label):
    # https://en.wikipedia.org/wiki/Help:Distinguishable_colors
    colors = """{240,163,255}	#F0A3FF	 	Amethyst
    {0,117,220}	#0075DC	 	Blue
    {153,63,0}	#993F00	 	Caramel
    {76,0,92}	#4C005C	 	Damson
    {0,92,49}	#005C31	 	Forest
    {43,206,72}	#2BCE48	 	Green
    {255,204,153}	#FFCC99	 	Honeydew
    {128,128,128}	#808080	 	Iron
    {148,255,181}	#94FFB5	 	Jade
    {143,124,0}	#8F7C00	 	Khaki
    {157,204,0}	#9DCC00	 	Lime
    {194,0,136}	#C20088	 	Mallow
    {0,51,128}	#003380	 	Navy
    {255,164,5}	#FFA405	 	Orpiment
    {255,168,187}	#FFA8BB	 	Pink
    {66,102,0}	#426600	 	Quagmire
    {255,0,16}	#FF0010	 	Red
    {94,241,242}	#5EF1F2	 	Sky
    {0,153,143}	#00998F	 	Turquoise
    {224,255,102}	#E0FF66	 	Uranium
    {116,10,255}	#740AFF	 	Violet
    {153,0,0}	#990000	 	Wine
    {255,255,128}	#FFFF80	 	Xanthin
    {255,255,0}	#FFFF00	 	Yellow
    {255,80,5}	#FF5005	 	Zinnia"""

    colors = re.findall(r"\d+,\d+,\d+", colors)
    colors = [[int(k) for k in c.split(",")] for c in colors]

    num_classes = len(num2label.keys())
    num2color = dict(zip(num2label.keys(), colors[:num_classes]))
    num2color[0] = [25, 25, 25]
    num2color[255] = [255, 255, 255]
    return num2color


def color_mask(mask, num2color):
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(np.uint8)
    for j in num2color.keys():
        colored_mask[mask == j] = num2color[j]
    return colored_mask


def visualize(num2color, **images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        if name == "mask":
            image = color_mask(image, num2color)
        plt.imshow(image)
    plt.show()
