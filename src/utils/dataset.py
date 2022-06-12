import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .augmentation import copyblob

class VOC_ProbDataset(Dataset):
    def __init__(
            self, 
            meta_df,
            augmentation=None,
            preprocessing=None,
            copypaste_prop=None,
            upsample_list=None,
            pre_aug=None,
            post_aug=None,
            label2num=None,
            vc_col=None,
            vc_df=None,
    ):
        self.ids = meta_df['id']
        self.images = meta_df['image_loc']
        self.masks = meta_df['mask_loc']
        self.meta_df = meta_df
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.copypaste_prop = copypaste_prop
        self.upsample_list = upsample_list
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        self.label2num = label2num

        self.vc_col = vc_col
        self.vc_df = vc_df        

    def __getitem__(self, i):
        # read data
        if self.vc_col:
            chosen = random.choices(self.vc_df[self.vc_col], self.vc_df['weight'], k=1)[0]
            idxes = self.meta_df[self.meta_df[self.vc_col] == chosen].index
            cur_id = random.choice(list(idxes))
            row = self.meta_df[self.meta_df.index == cur_id]
            
            dst_id = row['id'].values[0]
            dst_img = np.array(Image.open(row['image_loc'].values[0]))
            dst_mask = np.array(Image.open(row['mask_loc'].values[0]))
        else:
            dst_id =  self.ids.iloc[i]
            dst_img = np.array(Image.open(self.images.iloc[i]))
            dst_mask = np.array(Image.open(self.masks.iloc[i]))

        if self.copypaste_prop:
            paste_prob = random.random()
            if paste_prob < self.copypaste_prop:
                result, image, mask = copyblob(dst_img, dst_mask, 
                                               self.meta_df, self.upsample_list, 
                                               self.label2num, self.pre_aug, self.post_aug)
                if result:
                    dst_img, dst_mask= image, mask
                    if self.preprocessing:
                        sample = self.preprocessing(image=dst_img, mask=dst_mask)
                        dst_img, dst_mask = sample['image'], sample['mask']
                    return dst_img, dst_mask, dst_id

        if self.augmentation:
            sample = self.augmentation(image=dst_img, mask=dst_mask)
            dst_img, dst_mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=dst_img, mask=dst_mask)
            dst_img, dst_mask = sample['image'], sample['mask']
        return dst_img, dst_mask, dst_id

    def __len__(self):
        return len(self.ids)


class VOCDataset(Dataset):
    def __init__(
            self, 
            meta_df, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = meta_df['id']
        self.images = meta_df['image_loc']
        self.masks = meta_df['mask_loc']
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images.iloc[i]))
        mask = np.array(Image.open(self.masks.iloc[i]))
        cur_id = self.ids.iloc[i]
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask, cur_id
        
    def __len__(self):
        return len(self.ids)

class VOC_CopyPasteDataset(Dataset):
    def __init__(
            self, 
            meta_df,
            augmentation=None,
            preprocessing=None,
            copypaste_prop = 0.5,
            upsample_list = None,
            pre_aug=None,
            post_aug=None,
            label2num=None
    ):
        self.ids = meta_df['id']
        self.images = meta_df['image_loc']
        self.masks = meta_df['mask_loc']
        self.meta_df = meta_df
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.copypaste_prop = copypaste_prop
        self.upsample_list = upsample_list
        self.pre_aug = pre_aug
        self.post_aug = post_aug
        self.label2num = label2num
        
    def __getitem__(self, i):
        # read data
        dst_img = np.array(Image.open(self.images.iloc[i]))
        dst_mask = np.array(Image.open(self.masks.iloc[i]))
        cur_id = self.ids.iloc[i]

        paste_prob = random.random()
        if paste_prob < self.copypaste_prop:
            result, image, mask = copyblob(dst_img, dst_mask, 
                                           self.meta_df, self.upsample_list, 
                                           self.label2num, self.pre_aug, self.post_aug)
            if result:
                dst_img, dst_mask= image, mask
                if self.preprocessing:
                    sample = self.preprocessing(image=dst_img, mask=dst_mask)
                    dst_img, dst_mask = sample['image'], sample['mask']
                return dst_img, dst_mask, cur_id

        if self.augmentation:
            sample = self.augmentation(image=dst_img, mask=dst_mask)
            dst_img, dst_mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=dst_img, mask=dst_mask)
            dst_img, dst_mask = sample['image'], sample['mask']
        return dst_img, dst_mask, cur_id

    def __len__(self):
        return len(self.ids)
