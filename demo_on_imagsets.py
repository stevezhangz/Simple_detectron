import torchvision
import torch
from methods import *
from dataset import *
from grad_detectron import *
from utils import *
from annotations import *

if __name__ == '__main__':
    device='cuda'
    random_seed=33
    img_size=412
    filter=0.28
    json_path = "/home/stevezhangz/to/coco/annotations/instances_train2017.json"
    img_path = "/home/stevezhangz/to/coco/train2017"
    #initialize detectron
    detectron=grad_clustering_detectron(layer_name='layer4',
                                        num_targets=3,
                                        device=device,
                                        threeshold=filter)
    # set up random seed
    setup_seed(random_seed)
    # load the dataset
    loader = load_COCO(json_path, img_path, Batch_size=10, img_size=img_size)
    for seq, (img, image_id, cls, bbox, orig_size) in enumerate(loader):
        img=img.to(device)
        out=detectron(img)
        show_obj(img,out,index=seq,cls_dict=imagenet_key)
        if seq==10:
            break