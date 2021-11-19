import torch
from torch import nn
from utils import *




def iou_matched_loss(detectron_pre,bbox,cls,loss_f=nn.CrossEntropyLoss()):
    """
    :param detectron_pre: [[[logits,bbox],[logits,bbox],[logits,bbox]],[[logits,bbox],[logits,bbox],[logits,bbox]]]
    :param bbox:
    :param cls:
    :return:
    """
    def max_matched(dt_boxes,gt_boxes,cls_,threeshold=0.4):
        logits_record={}
        for seq_dt,dt_box in enumerate(dt_boxes):
            current_max_id=0
            iou_max=-1000
            for seq_gt,gt_box in enumerate(gt_boxes):
                iou=iou_compute(dt_box[1],gt_box)
                if iou>iou_max:
                    iou_max=iou
                    current_max_id=seq_gt
            if iou_max>=threeshold:
                logits_record[str(dt_box[1])]=(seq_dt,cls_[current_max_id])
        return logits_record

    def all_targets_matched(target_dt_boxes,train_gt_boxes):
        all_logits=[]
        for target_id in range(len(target_dt_boxes)):
            target_dt_box,target_gt_boxes=target_dt_boxes[target_id],train_gt_boxes[target_id]
            logits=max_matched(target_dt_box,target_gt_boxes,cls[target_id])
            all_logits.append(logits)
        return all_logits

    all_logits=all_targets_matched(detectron_pre,bbox)

    loss=0
    for target_id in range(len(all_logits)):
        current_gt_clses=cls[target_id]
        current_target_logits=all_logits[target_id]

        current_target_pre=detectron_pre[target_id]
        logits_all = []
        y_all = []
        for box in current_target_pre:
            if str(box[1]) in current_target_logits:
                seq_dt,logits=current_target_logits[str(box[1])]
                y_all.append(logits)
                logits_all.append(current_target_pre[seq_dt][0].reshape(1,-1))
        if logits_all!=[]:
            device=current_target_pre[seq_dt][0].device
            loss=loss_f(torch.cat(logits_all,dim=0),torch.from_numpy(np.array(y_all)).to(device))


            return loss
        else:
            return None



