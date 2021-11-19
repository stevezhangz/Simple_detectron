import torch
from torch import nn
import numpy as np
from utils import *
from methods import *
import torchvision

class back_forward_record(object):
    def __init__(self,model,layer_name,num_targets=1,orig_size=(412,412),pattern='GAP'):
        super().__init__()
        assert pattern in ['GAD','GAP'],print('only support grad or gap')
        self.forward_hook=list()
        self.backward_hook=list()
        self.model=model
        self.model.eval()
        self.num_targets=num_targets
        self.origi_size=orig_size
        self.pattern=pattern

        for name,sublayer in model.named_children():
            if name==layer_name:
                sublayer.register_backward_hook(self.grad_record)
                sublayer.register_forward_hook(self.feature_record)

    def feature_record(self,module,input,output):
        self.forward_hook.append(output)

    def grad_record(self,module,input,output):
        self.backward_hook.append(output[0])

    def forward(self,images):
        return self.model(images)

    def backward(self,one_hot):
        self.model.backward(one_hot)

class GRAD_detectron(back_forward_record):

    def back_progation(self,images):
        pre=self.forward(images)
        one_hot_pres=mask_one_hot(pre,topk=self.num_targets)
        return pre,one_hot_pres

    def cam_compute(self,pre,one_hot):
        pre.backward(one_hot,retain_graph=True)
        image_size=self.forward_hook[0][0].size()[2:]
        weight = nn.AvgPool2d(image_size)(l2_normalize(self.backward_hook[0])).squeeze(-1).squeeze(-1)
        if self.pattern=='GAD':
            weight = nn.Softmax(dim=-1)(weight)
        weight = weight.unsqueeze(-1).unsqueeze(-1)
        Cam_r = torch.zeros((self.forward_hook[0].shape[0], self.forward_hook[0].shape[2], self.forward_hook[0].shape[3]))
        for i in range(weight.size()[0]):
            for w, f in zip(weight[i], self.forward_hook[0][i]):
                weight_=w.expand_as(f)
                Cam_r[i] += weight_.cpu()* f.cpu()
            Cam_r[i] = nn.ReLU()(Cam_r[i])
            Cam_r[i] = max_min_normalize(Cam_r[i])
        Cam_r = Cam_r.cpu().detach().numpy()
        return Cam_r

    def grad2targetloaction(self, cam_array, threeshold):
        obj_record, trace, cam_array = depict_obj(cam_array, threeshold)
        return obj_record

    def scale(self,array,current_size,orig_size):
        return map2original_size(array,current_size,orig_size)

    def grad_compute(self,images,threeshold):
        target_predict=[]
        pre,one_hot_pres=self.back_progation(images)
        for one_hot_pre in one_hot_pres:
            cam=self.cam_compute(pre,one_hot_pre)
            key=one_hot_pre.argmax(-1).cpu().detach().numpy()
            btz_record={}
            for id_target in range(key.shape[0]):
                key_, cam_array=key[id_target],cam[id_target]
                obj_higlight=self.grad2targetloaction(cam_array,threeshold)
                highlight_obj=outline_obj(obj_higlight,cam_array)
                btz_record[id_target]=(key_,self.scale(highlight_obj,(cam.shape[-2],cam.shape[-1]),self.origi_size))
            target_predict.append(btz_record)

        post_process={}

        for seqential_pre in range(len(target_predict)):

            for target_id in target_predict[seqential_pre]:
                if target_id not in post_process:
                    post_process[target_id]=[]
                    post_process[target_id].append(target_predict[seqential_pre][target_id])
                else:
                    post_process[target_id].append(target_predict[seqential_pre][target_id])

        return target_predict

