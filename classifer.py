from torch import nn
import torch

class Classifier(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        hidden=int(in_feature/2)
        self.cls_layer = nn.Sequential(nn.Linear(in_features=in_feature, out_features=hidden),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=hidden, out_features=hidden),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(in_features=hidden, out_features=out_feature))

    def __call__(self, image):
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
        x = self.avg(image).flatten(1)
        y = self.cls_layer(x)
        return y

class Classifier2(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_layer = nn.Sequential(
            nn.Linear(4,out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=500,out_features=500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=500, out_features=out_feature),
        )


    def __call__(self, image):
        if len(image.size()) == 3:
            image = image.unsqueeze(0)
        device=image.device
        _,ch,w,h=image.size()
        mid_w=int(w/2)
        mid_h=int(h/2)
        roi=torch.zeros(size=(_,ch,2,2)).to(device)
        roi[:, :, 0, 0] += nn.AdaptiveMaxPool2d((1, 1))(image[:, :, 0:mid_w, 0:mid_h]).flatten(-3)
        roi[:,:,0,1]+=nn.AdaptiveMaxPool2d((1, 1))(image[:,:,mid_w:w,0:mid_h]).flatten(-3)

        roi[:,:,1,0]+= nn.AdaptiveMaxPool2d((1, 1))(image[:, :, 0:mid_w, mid_h:h]).flatten(-3)
        roi[:, :, 1, 1]+= nn.AdaptiveMaxPool2d((1, 1))(image[:, :, mid_w:w, mid_h:h]).flatten(-3)


        y = self.cls_layer(torch.mean(roi,dim=1).flatten(1))
        return y