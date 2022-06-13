import cv2
import numpy as np
import random
import albumentations as albu
from albumentations.core.transforms_interface import DualTransform, to_tuple
from PIL import Image
# https://github.com/VainF/DeepLabV3Plus-Pytorch/blob/master/main.py#L99

class RandomScale(DualTransform):
    def __init__(self, scale_limit=0.1, interpolation=cv2.INTER_LINEAR, always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)
        self.scale_limit = to_tuple(scale_limit, bias=-1.0)
        self.interpolation = interpolation

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv2.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        new_h = int((1+scale) * h)
        new_w = int((1+scale) * w)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def apply_to_mask(self, img, scale=0, interpolation=cv2.INTER_NEAREST, **params):
        h, w = img.shape[:2]
        new_h = int((1+scale) * h)
        new_w = int((1+scale) * w)        
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def get_training_augmentation(scale_limit=(0.8, 2.0)):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        RandomScale(scale_limit=scale_limit, p=1),
        albu.ShiftScaleRotate(scale_limit=0., rotate_limit=15, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=512, min_width=512,
                        always_apply=True, border_mode=0,
                        mask_value=0),
        albu.RandomCrop(512, 512, p=1),
        albu.RandomBrightnessContrast(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.GaussNoise(p=1),        
                albu.RandomGamma(p=1),
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(min_height= None, min_width= None,
                        pad_height_divisor= 32, pad_width_divisor=32,
                         always_apply=True, border_mode=0,
                        mask_value=0),
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor),
    ]
    return albu.Compose(_transform)

def get_preaug(scale_limit):
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        RandomScale(scale_limit=scale_limit, p=1),
        albu.ShiftScaleRotate(scale_limit=0., rotate_limit=15, p=1, border_mode=0),
    ]
    return albu.Compose(train_transform)

def get_postaug():
    train_transform = [
        albu.PadIfNeeded(min_height=512, min_width=512,
                                always_apply=True, border_mode=0,
                                mask_value=0),
        albu.RandomCrop(512, 512, p=1),
        albu.RandomBrightnessContrast(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.GaussNoise(p=1),        
                albu.RandomGamma(p=1),
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)

def copyblob(dst_img, dst_mask, meta_df, paste_list, label2num, get_preaug, get_postaug):
    src_label = random.choice(paste_list)
    src_df = meta_df[meta_df[f'{src_label}_cnt'] == True]
    src_data_num =  random.randint(0,len(src_df)-1)
    src_img = np.array(Image.open(src_df.iloc[src_data_num]['image_loc']))
    src_mask = np.array(Image.open(src_df.iloc[src_data_num]['mask_loc']))

    src_class = label2num[src_label]
    dst_class = label2num['background']

    sample_dst = get_preaug(scale_limit=(0.8, 2.0))(image=dst_img, mask=dst_mask)
    dst_img, dst_mask = sample_dst['image'], sample_dst['mask']
    
    pre_src_cnt = np.sum(src_mask == src_class) / (src_mask.shape[0] * src_mask.shape[1])
    if pre_src_cnt < 0.1:
        sample_src = get_preaug(scale_limit=(1.0, 2.0))(image=src_img, mask=src_mask)
    else:
        sample_src = get_preaug(scale_limit=(0.5, 1.5))(image=src_img, mask=src_mask)
    
    src_img, src_mask = sample_src['image'], sample_src['mask']
    
    src_cnt = np.sum(src_mask==src_class)
    dst_cnt = np.sum(dst_mask==dst_class)

    if src_cnt != 0 and dst_cnt != 0:
        src_idx = np.where(src_mask==src_class)

        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)        
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]

        dst_idx = np.where(dst_mask==dst_class)
        topleft_idx = np.where((dst_idx[0] <= np.percentile(dst_idx[0], 50)) &
                           (dst_idx[1] <= np.percentile(dst_idx[1], 50)))
        dst_idx = (dst_idx[0][topleft_idx], dst_idx[1][topleft_idx])

        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx] 

        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))

        avail_idx = np.where((dst_idx[0] < dst_mask.shape[0]) &
                             (dst_idx[1] < dst_mask.shape[1]) &
                             (dst_idx[0] >= 0) &
                             (dst_idx[1] >= 0))
        dst_idx = (dst_idx[0][avail_idx], dst_idx[1][avail_idx])
        src_idx = (src_idx[0][avail_idx], src_idx[1][avail_idx])

        dst_mask[dst_idx] = src_class
        dst_img[dst_idx[0], dst_idx[1]] = src_img[src_idx[0], src_idx[1]]

        mask_pasted = (dst_mask == src_class).astype(np.uint8)
        mask_pasted_idx = np.where(mask_pasted == 1)
        kernel = np.ones((7,7), np.uint8)
        dilated = cv2.dilate(mask_pasted, kernel, iterations = 1)
        dilated[mask_pasted_idx] = 0
        dilated_idx = np.where(dilated == 1)
        dst_mask[dilated_idx] = 255
        sample_dst = get_postaug(image=dst_img, mask=dst_mask)
        dst_img, dst_mask = sample_dst['image'], sample_dst['mask']
        return True, dst_img, dst_mask
    else:
        return False, None, None

