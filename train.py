import torchvision
import torch
from methods import *
from dataset import *
from grad_detectron import *
from utils import *
from classifer import *

def train(json_path = "/home/stevezhangz/to/coco/annotations/instances_train2017.json",
          img_path = "/home/stevezhangz/to/coco/train2017",
          model=Classifier2(in_feature=100,out_feature=91),
          extractor=model_rebuild(resnet(layers='50')),
          epoch=2,
          learning_rate=0.001,
          save_dir='cls.pth',
          loss_f=torch.nn.NLLLoss(),
          save_freq=1000,
          show_freq=1,
          device='cuda',
          img_size=412,
          random_seed=33,
          max_per_box_num=4,
          state_dict_dir=None):
    """
    This method is used to train the linear classifier, based on ROI-likely methods
    :param json_path:  coco annotation path
    :param img_path: coco dataset path
    :param model: the convolution model you want to train
    :param epoch: trainning epoch
    :param learning_rate:
    :param save_dir: save the model's parameter to this dir
    :param loss_f: loss function
    :param save_freq: save model's parameter save_freq once time
    :param device: cpu or gpu
    :param img_size: reshape the crouped image to this size
    :param random_seed: random_seed for pytorch and numpy
    :param max_per_box_num: maximum of the crouped images of specific image
    :param state_dict_dir: pre-trained pth
    """


    # ConvNet of ResNet was responsible for
    extractor.eval()
    if device:
        extractor.to(device)
        model.to(device)
    # set up random seed for numpy and tensorflow
    setup_seed(random_seed)
    # load the dataset
    loader = load_COCO(json_path, img_path, Batch_size=2, img_size=img_size)
    # load pre-train parameters
    if state_dict_dir!=None:
        if os.path.exists(state_dict_dir):
            model.load_state_dict(torch.load(state_dict_dir)['model'])
            optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
            optim.load_state_dict(torch.load(state_dict_dir)['optim'])
    else:
        optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    # max_train_iter: you know for each image there maybe more than one bboxes, so it's a huge GPU consumption.
    # So we have to introduce a limit for the num of bbox target to be trained.
    max_train_iter=max_per_box_num

    for epc in range(epoch):
        # used for recording the top1-acc and so on
        total=0
        top1=0
        top5=0
        # img, img_id, cls, bbox, orig_size
        for seq, (img, image_id, cls, bbox,orig_size) in enumerate(loader):
            all_logits=[]
            y=[]
            for img_id in range(img.size()[0]):
                for seq1,bbox_per in enumerate(bbox[img_id]):
                    if seq1>=max_train_iter:
                        continue
                    current_box_cls=cls[img_id][seq1]
                    feature_map=extractor(img[img_id].unsqueeze(0).to(device)).squeeze(0)
                    channels,f_w,f_h=feature_map.size()
                    bbox_roi=xywh2xyxy(bbox_per)
                    ratio=[f_w/img_size,f_h/img_size,f_w/img_size,f_h/img_size]
                    bbox_roi=[int(bbox_roi[i]*ratio[i]) for i in range(len(ratio))]
                    try:
                        # roi proposal
                        croped_image=sub_crop_resize(feature_map,bbox_roi,size=(f_w,f_h))
                        all_logits.append(model(croped_image.unsqueeze(0).to(device)).to('cpu'))
                        y.append(current_box_cls)
                    except:
                        continue

            if all_logits!=[]:
                y=torch.from_numpy(np.array(y)).to(device)
                pre=torch.cat(all_logits,dim=0).to(device)
                loss=loss_f(pre.softmax(-1),y)

                # accuracy record
                total += pre.size()[0]
                test_labels = y.view(-1, 1)  # reshape labels from [n] to [n,1] to compare [n,k]
                _, max1 = torch.topk(pre, 1, dim=-1)
                _, max5 = torch.topk(pre, 5, dim=-1)
                top1 += (test_labels == max1[:, 0:1]).sum().item()
                top5 += (test_labels == max5).sum().item()

                # optimize
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)
                optim.step()

                # condition for saving
                if seq % show_freq == 0 and seq >= show_freq:
                    print(f'|epc-{epc}-iter-{seq + 1}-training loss is-{loss.cpu().detach().numpy()}-top1_acc-{top1 / total}-top5_acc-{top5 / total}|')

                if seq%save_freq==0 and seq >=save_freq:
                    torch.save({'model':model.state_dict(),"optim":optim.state_dict()},save_dir)
                    top1=0
                    top5=0
                    total=0

if __name__=="__main__":
    train()





