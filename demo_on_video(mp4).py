import cv2
from grad_detectron import *
from utils import *
from annotations import *

img_size=412
video_dir='car.mp4'
device='cuda'
random_seed=33
filter=0.43
get_video= cv2.VideoCapture(video_dir)
# set up random seed

setup_seed(random_seed)
fps = get_video.get(cv2.CAP_PROP_FPS)
size = (img_size,img_size)
detectron=grad_clustering_detectron(layer_name='layer4',
                                    num_targets=2,
                                    device=device,
                                    threeshold=filter,
                                    orig_size=size)
fums = get_video.get(cv2.CAP_PROP_FRAME_COUNT)
t, frame = get_video.read()

while t:
    frame=cv2.resize(frame,size)
    tensor=T.Compose([
        T.ToTensor(),
        T.Resize(size)
    ])(frame)
    # get bbox and class
    pre=detectron(tensor.unsqueeze(0).to(device))
    # get annotated image
    frame=video_capture_bbox(frame,pre,cls_dict=imagenet_key)


    #show video
    cv2.imshow('video', frame)
    print(1)
    cv2.waitKey(int(1000/ int(fps)*1.5))
    t, frame = get_video.read()
get_video.release()