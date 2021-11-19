import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
from torchvision.transforms import transforms
import os

def load_coco_datadir(json_path,img_dir):
    """
    return [
    [img_id,img_dir]
    ]
    """
    def path_join(home_dir,son_dir):
        assert os.path.exists(home_dir),print(home_dir+" not exists")
        pth=os.path.join(home_dir,son_dir)
        return pth
    info = COCO(annotation_file=json_path)
    info_keys=info.imgs.keys()# list type
    img_set_dir=[]
    for info_key in info_keys:
        pth=path_join(img_dir, info.imgs[info_key]['file_name'])
        if os.path.exists(pth):
            annIds = info.getAnnIds(imgIds=info.imgs[info_key]['id'])
            anns = info.loadAnns(annIds)
            bbox=[]
            cls=[]
            for i in anns:
                bbox.append(i["bbox"])
                cls.append(i["category_id"])
            img_set_dir.append((pth, info.imgs[info_key]['id'],cls,bbox,(info.imgs[info_key]['width'],info.imgs[info_key]['height'])))
        else:
            continue
    return img_set_dir

class my_dataset(torch.utils.data.Dataset):

    def __init__(self,data_pth,transform):
        self.root_dir=data_pth
        self.transform=transform
    def __len__(self):
        return len(self.root_dir)
    def __getitem__(self, item):
        (img_dir,img_id,cls,bbox,origi_size)=self.root_dir[item]
        img=Image.open(img_dir)
        img=np.array(img)
        img=self.transform(img)
        if img.size()[0]==1:
            img=img.repeat(3,1,1)
        c,w,h=img.size()
        # transform bbox to specific size
        shaped_bbox=[]
        for per_box in bbox:
            shaped_bbox.append([per_box[0]/(origi_size[0]+1e-3)*w,
                                   per_box[1]/(origi_size[1]+1e-3)*h,
                                   per_box[2]/(origi_size[0]+1e-3)*w,
                                   per_box[3]/(origi_size[1]+1e-3)*h])
        del (bbox)
        return img,img_id,cls,shaped_bbox,origi_size

def lable_annotations(labels,total_cls=91):
    btz_l=len(labels)
    one_hot=np.zeros(shape=(btz_l,total_cls))
    for seq,val in enumerate(labels):
        one_hot[seq,val]=1
    return one_hot

def one_hot_collate_fn(batch):
    size=len(batch)
    batch = list(zip(*batch))
    img,img_id,cls,bbox,orig_size=batch[0],[*batch[1]],[*batch[2]],[*batch[3]],[*batch[4]]
    processed_cls=[]
    for cls_ in cls:
        processed_cls.append(lable_annotations(cls_))
    del(cls)
    del(batch)
    new_img=[]
    for im in img:
        if len(im.size())==3:
          new_img.append(im.unsqueeze(0))
        else:
            new_img.append(im)
    img=new_img
    return torch.cat(img,dim=0),img_id,processed_cls,bbox,orig_size

def collate_fn(batch):
    size=len(batch)
    batch = list(zip(*batch))
    img,img_id,cls,bbox,orig_size=batch[0],[*batch[1]],[*batch[2]],[*batch[3]],[*batch[4]]

    del(batch)
    new_img=[]
    for im in img:
        if len(im.size())==3:
          new_img.append(im.unsqueeze(0))
        else:
            new_img.append(im)
    img=new_img
    return torch.cat(img,dim=0),img_id,cls,bbox,orig_size

def load_COCO(home_dir,son_dir,transform=None,Batch_size=15,img_size=412):
    """
    in order to evaluate the Dt, we need img_id, so this method will return the imgset, img_id
    :param home_dir: dir of img folder
    :param son_dir: son pth of img
    :param transform: data process method from "torchvision.transforms "
    :param Batch_size:
    :param img_size:
    :return: data_btz
    """
    if transform==None:
        pattern = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(img_size,img_size))
        ])
    data_pth=load_coco_datadir(home_dir,son_dir)
    train_set=my_dataset(data_pth,transform=pattern)
    train_loader = DataLoader(
        train_set,
        batch_size=Batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    return train_loader


