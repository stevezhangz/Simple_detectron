import torch
import numpy as np
import copy
import torchvision.transforms.transforms as T
from PIL import ImageDraw
import random
import os
import cv2


def mask_one_hot(pre,topk):
    """
    according the max topk prediction to create the one-hot matrices
    :param pre: prediction of convolution nn
    :param topk: number of topk prediction

    :return: masked one-hot prediction
    """
    sorted_logits,indices=torch.sort(pre.softmax(-1),dim=-1,descending=True)
    pres=[]
    for i in range(topk):
        one_hot_matrice = torch.zeros_like(pre)
        for btz in range(one_hot_matrice.shape[0]):
            one_hot_matrice[btz,indices[btz,i]]=1
        pres.append(one_hot_matrice)
    return pres

def l2_normalize(tensors,epsilon=1e-3):
    '''
    divide the matrice by its own l2 norm
    :param tensors: input
    :param epsilon: avoid the numerical overflow
    :return:
    '''
    return tensors/(torch.norm(tensors,p=2)**0.5+epsilon)

def max_min_normalize(tensors,epsilon=1e-3):
    """
    a way of normalize
    :param tensors:
    :param epsilon:
    :return:
    """
    return (tensors-tensors.min())/(tensors.max()+epsilon)

def _mask(matrices,threeshould,fill=None):
    '''
    mask the value which lower than the threshold
    :param matrices:
    :param threeshould:
    :param fill:
    :return:
    '''
    assert isinstance(matrices,np.ndarray),print('please use numpy form input instead')
    if fill!=None:
        zero_matrices=fill
    else:
        zero_matrices=np.zeros_like(matrices)
    new_matrice=np.where(matrices>threeshould,matrices,zero_matrices)
    return new_matrice

def broad_product(mat1,mat2):
    """
    if u waht to use a weighted matrice to multiply the target matrices, it's a good method
    """
    assert mat1.size()[-1]==1 or mat2.size()[-1]==1,print('couldnot broad')
    if mat1.size()[-1]==1:
        return mat1.expand_as(mat2)*mat2
    else:
        return mat2.expand_as(mat1)*mat1

def per_contact(dict1, dict2):
    """
    move specific item from dict2 to dict1, which has the key of dict1
    """
    dict2_key = list(dict2.keys())
    for i in dict1:
        if i in dict2_key:
            for element in dict2[i]:
                dict1[i].append(element)
    return dict1

def dicts_contact(dicts):
    """
    [dict1,dict2,dict3]
    if u has a list of dicts, you can combine the above method with current method to contact
    """
    dict1=dicts[0]
    for dict_ in dicts[1:]:
        dict1=per_contact(dict1,dict_)
    return dict1

def iou_compute(outline1,outline2):
    def area_compute(outline):
        """
        count the area size of region of 'outline'
        """
        w=outline[2]-outline[0]
        h=outline[3]-outline[1]

        if w>0 and h>0:
            return w*h
        else:
            return -w*h
    """
    PS: only support the xyxy type
    principle: maximum of the left down point and minimum of  the right up point
    """
    outline2=xywh2xyxy(outline2)
    inner_outline=[max(outline1[0],outline2[0]),max(outline1[1],outline2[1]),min(outline1[2],outline2[2]),min(outline1[3],outline2[3])]
    inner_area=area_compute(inner_outline)
    area1=area_compute(outline1)
    area2=area_compute(outline2)

    return inner_area/(area1+area2-inner_area)

def xywh2xyxy(outline):
    """
    transform [xywh] 2 [xyxy]
    """
    return [outline[0],outline[1],outline[0]+outline[2],outline[3]+outline[1]]

def xyxy2xywh(outline):
    """
    transform [xyxy] 2 [xywh]
    """
    w=outline[2]-outline[0]
    h=outline[3]-outline[1]
    return [outline[0],outline[1],w,h]


def sub_crop_resize(tensor,outline,size=None,retain_orig_size=None):
    """
    crop sub-tensor from tensor based on the bbox of outline.
    then resized the croped-tensor to 'size'
    """
    if size!=None:
        assert isinstance(size,tuple)
        w,h=size
    outline = [min(max(0, outline[0]), w - 1), min(max(0, outline[1]), h - 1), min(max(0, outline[2]), w - 1),
                min(max(0, outline[3]), h - 1)]
    outline=[int(i) for i in outline]
    if outline[0]==outline[2]:
        if outline[0]==w-1:
            outline[0]-=1
        elif outline[2]==0:
            outline[2]+=1
        else:
            outline[1]-=1

    if outline[1]==outline[3]:
        if outline[3]==h-1:
            outline[1]-=1
        elif outline[1]==0:
            outline[3]+=1
        else:
            outline[1]-=1
    if retain_orig_size!=None:
        assert isinstance(retain_orig_size,tuple)
        sub_tensor=copy.deepcopy(tensor.cpu().detach())
        sub_tensor=T.ToPILImage()(sub_tensor[:, outline[0]:outline[2], outline[1]:outline[3]])
        return T.Compose([
            T.ToTensor(),
            T.Resize((w,h))
        ])(sub_tensor)
    else:
        return copy.deepcopy(tensor[:, outline[0]:outline[2], outline[1]:outline[3]].detach().cpu())


def sub_crops_resize(tensor,outlines,size):
    """
    outlines:{
    'cls1':[bbox1,bbox2,etc.]

    }
    crop sub-tensor from tensor based on a series bboxes of outline.
    then resized the croped-tensor to 'size'
    """
    assert isinstance(size,tuple),print('size should be tuple dtype such as (x,y)')
    new_outlines={}
    for cls in outlines:
        left_down_x=100000
        left_down_y=100000
        right_up_x=0
        right_up_y=0
        if outlines[cls]!=[]:
            for bbox in outlines[cls]:
                if bbox[0]<left_down_x:
                    left_down_x=bbox[0]
                if bbox[1]<left_down_y:
                    left_down_y=bbox[1]
                if bbox[2]>right_up_x:
                    right_up_x=bbox[2]
                if bbox[3]>right_up_y:
                    right_up_y=bbox[3]
            try:
                new_outlines[cls]=sub_crop_resize(tensor,[left_down_x,left_down_y,right_up_x,right_up_y],size=size)
            except:
                print(f'failed bbox:{[left_down_x,left_down_y,right_up_x,right_up_y]}')
    return new_outlines


def show_obj(imageset,objrecord,outline_color='random',width=2,index=None,cls_dict=None):
    """
    papers:
        imageset: input tensor
        objrecord: output of 'consciousness_matched_detectron'
        outline_color: color of the box, if this choice is random, then randomly choice a color from ['red','blue','purple','black','orange']
        width: width of bbox
    """
    color=outline_color
    for obj_index in objrecord:
        #current_img=imageset[obj_index]
        current_img=T.ToPILImage()(imageset[obj_index]).getdata()
        cv2img=cv2.cvtColor(np.uint8(current_img).reshape((412,412,3)),cv2.COLOR_BGR2RGB)
        record_site = []
        for target_outline in objrecord[obj_index]:
            for seq_outline,outline in enumerate(target_outline[1]):
                location=(outline[0]+20,outline[1]+20)
                while(location in record_site):
                    location=(record_site[record_site.index(location)][0],record_site[record_site.index(location)][1]+20)
                    if location not in record_site:
                        break
                record_site.append(location)
                if outline_color == 'random':
                    color=random.choice([(0,255,0),(0,0,255),(255,0,0)])
                if cls_dict!=None:
                    text = str(cls_dict[target_outline[0]])
                else:
                    text=cls_dict[target_outline[0]]
                if not isinstance(text,str):
                    text=str(text)
                cv2.rectangle(cv2img,(outline[0],outline[1]),(outline[2],outline[3]),color, width)
                cv2.putText(cv2img, text,location , cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            if not os.path.exists('img_save'):
                os.mkdir('img_save')
            cv2.imwrite('img_save/'+f'{index}'+ f'{obj_index}.jpg',cv2img)
def video_capture_bbox(tensor,outline,outline_color='random',width=2,cls_dict=None):
    objrecord=outline[0]
    record_site = []
    for target_outline in objrecord:
        for seq_outline, outline in enumerate(target_outline[1]):
            location = (outline[0] + 20, outline[1] + 20)
            while (location in record_site):
                location = (
                record_site[record_site.index(location)][0], record_site[record_site.index(location)][1] + 20)
                if location not in record_site:
                    break
            record_site.append(location)
            if outline_color == 'random':
                color = random.choice([(0, 255, 0), (0, 0, 255), (255, 0, 0)])
            if cls_dict != None:
                text = str(cls_dict[target_outline[0]])
            else:
                text = cls_dict[target_outline[0]]
            if not isinstance(text, str):
                text = str(text)
            cv2.rectangle(tensor, (outline[0], outline[1]), (outline[2], outline[3]), color, width)
            cv2.putText(tensor, text, location, cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return tensor
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

