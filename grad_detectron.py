import torch
from torch import nn
import torchvision
from methods import *
from utils import *
from grad_compute import *



def final_pre2cocodtype(final_pre,id):
    dt=[]
    for image_id in range(len(final_pre)):
        #{"image_id":42,"category_id":18,"bbox":[258.15,41.29,348.26,243.78],"score":0.236}
        for seq,pre in enumerate(final_pre[image_id]):
            dt_per = {}
            dt_per["image_id"] =int(id[image_id])
            dt_per["category_id"]=int(pre[0].reshape(1,-1).cpu().detach().softmax(-1).argmax(-1).numpy()[0])
            dt_per["bbox"]=pre[1]
            dt_per["score"]=float(pre[0].cpu().detach().softmax(-1).numpy()[0,dt_per["category_id"]])
            dt.append(dt_per)
    return dt



def resnet(layers='50'):
    """
    this method recall the method of torchvision.model,resnet
    it's used to facilitated the model build process
    """
    model = {
        '50':torchvision.models.resnet50(pretrained=True),
        '34':torchvision.models.resnet34(pretrained=True),
        '18':torchvision.models.resnet18(pretrained=True)
    }
    assert layers in model, print(f'only support fellow choices:{list(model.keys())}')
    return model[layers]

def vision_detection(vision_model,cropd_images,device=None):
    """
    the detectron could be split into two different parts, first is used to find the object, another is to judge which
    the object is.
    this method is to classify the detected object
    """
    if device !=None:
        vision_model=vision_model.to(device)
    total_r=[]
    for cropd_set in cropd_images:
        set_r=[]
        for crop_image in cropd_set:
            if device!=None:
                crop_image_=crop_image[0].to(device)
            else:
                crop_image_=crop_image[0]
            predictions=vision_model(crop_image_)
            del(crop_image_)
            set_r.append((predictions,crop_image[1]))
        total_r.append(set_r)
    return total_r

class model_rebuild(nn.Module):
    """
    the last dim of pre-trained resnet is 1000, but coco only has 91 objects id, so this method is used to replaced the
    original Linear layer by a re-build one.
    """
    def __init__(self,model):
        super().__init__()
        del(model.fc)
        self.head=nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu
        )
        self.layer1=model.layer1
        self.layer2=model.layer2
        self.layer3=model.layer3
        self.layer4=model.layer4
        self.g_avg=nn.AdaptiveAvgPool2d((1,1))

    def __call__(self, x):
        x1=self.head(x)
        x2=self.layer1(x1)
        x3=self.layer2(x2)
        x4=self.layer3(x3)
        x5=self.layer4(x4)
        return x5
class vgg16_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.CONV=torchvision.models.vgg16(pretrained=True).features
    def __call__(self,img):
        if len(img.size())!=4:
            img=img.unsqueeze(0)
        return self.CONV(img)
class grad_clustering_detectron(nn.Module):
    """
    if u have read my paper you can know what it is.
    """
    def __init__(self,
                 layer_name='layer4',
                 num_targets=1,
                 orig_size=412,
                 threeshold=0.4,
                 pattern="GAD",
                 consciousness_model=None,
                 device=None):
        super().__init__()
        if consciousness_model==None:
            self.consciousness_model=resnet()
        self.threeshold=threeshold
        self.consciousness_model.eval()
        self.device = device
        if device!=None:
            self.consciousness_model.to(device)
        self.GRAD_CAM = GRAD_detectron(model=self.consciousness_model,
                                       layer_name=layer_name,
                                       num_targets=num_targets,
                                       orig_size=orig_size,
                                       pattern=pattern)
    def __call__(self, img):
        target_predict = self.GRAD_CAM.grad_compute(img,threeshold=self.threeshold)
        self.GRAD_CAM.backward_hook=list() # release the space for grad recording
        self.GRAD_CAM.forward_hook=list()  # release the space for grad recording
        final_outlines = target_post(img, target_predict)
        return final_outlines

