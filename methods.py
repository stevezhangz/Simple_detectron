from utils import *
def gather_up_base_common_unit(lists):
    """
    for example:
    input:L=[
    [1,2,3,4,5,6],
    [2,9,10],
    [3,1000]
    ,[100000000000]
    ]
    output: [[1, 2, 3, 4, 5, 6, 2, 9, 10, 3, 1000], [100000000000]]
    """
    def overlap(m1, m2):
        for units in m2:
            if units in m1:
                for units in m2:
                    m1.append(units)
                return m1
        return []

    gather_up = []
    extra=[]
    for seq, val in enumerate(lists):
        if seq == 0:
            gather_up.append(val)
        else:
            for seq2, element in enumerate(gather_up):
                contact = overlap(element, val)
                if contact:
                    gather_up[seq2] = contact
                else:
                    extra.append(val)
    return gather_up+extra


def side_record(matrice,threeshould=0.3,display=True):
    '''
    input: grid matrice
    output:
       {
       1:(startpoint1, endpoint1),`````
       }
    '''
    assert isinstance(matrice,np.ndarray),print('plz use ndarray form input instead')
    W,H=matrice.shape
    line_record={}
    for i1 in range(W):
        start=0
        strin=''
        startnode=0
        line_record[i1]=[]
        for i2 in range(H):
            node=matrice[i1,i2]
            strin+=str(node)[:3]+'     '
            if isinstance(node,str):
                node=eval(node)
            if start:
                if node<=threeshould:
                    start=0
                    line_record[i1].append((startnode,max(i2-1,0)))
                elif i2==H-1 and node>threeshould:
                    line_record[i1].append((startnode,i2-1))
                else:
                    continue
            else:
                if node >threeshould:
                    start=1
                    startnode=i2
                if i2==W-1 and node>threeshould:
                    line_record[i1].append((i2-1,i2-1))
        if display:
            print(strin)
            print(line_record[i1])
    return line_record


def dynamic(obj_record, trace, matrice, x, y, threeshold):
    """
    Dynamic programming based method, combine with "depict_obj" to predict the obj
    :param obj_record: record the pointor coordinates of specific target
    :param trace: record whether a point has been searched
    :param matrice: gradient array
    :param x: current x coordinates
    :param y: current y coordinates
    :param threeshold: threshold of grad sinal
    :return: obj_record, trace, matrice
    """
    if (x, y) not in obj_record:
        obj_record.append((x, y))
    directions = [
        (x, y + 1),
        (x, y - 1),
        (x - 1, y),
        (x + 1, y)
    ]
    for direct in directions:
        if direct[0] < matrice.shape[0] and direct[1] < matrice.shape[1] and direct[0] >= 0 and direct[1] >= 0 and \
                trace[direct[0], direct[1]] != 1:
            if matrice[direct[0], direct[1]] >= threeshold:
                obj_record.append((direct[0], direct[1]))
                trace[direct[0], direct[1]] = 1
                obj_record, trace, matrice = dynamic(obj_record, trace, matrice, direct[0], direct[1], threeshold)

    return obj_record, trace, matrice


def depict_obj(matrice, threeshold):
    """
    combine with "dynamic" to predict the obj
    :param matrice:
    :param threeshold:
    :return: obj_record, trace, matrice
    """
    w, h = matrice.shape
    trace = np.zeros_like(matrice)
    obj_record = []
    for id_w in range(w):
        for id_h in range(h):
            if trace[id_w, id_h] != 1 and matrice[id_w, id_h] >= threeshold:
                obj_ = []
                obj_record_, trace, matrice = dynamic(obj_, trace, matrice, id_w, id_h, threeshold)
                if obj_record_ != []:
                    obj_record.append(obj_record_)
    return obj_record, trace, matrice


def outline_obj(obj_point,matrice,amplify_point_bbox=True):
    """
    based on searching method, dipict the object based on the gradient array.
    :param obj_point:
    :param matrice: feature map(np.ndarray type)
    :param amplify_point_bbox: based on clustering method, you may receive one point set, so creat a local rectangle could help save some objects.
    :return:
    """
    obj_r=[]
    for point_set in obj_point:
        width,length=matrice.shape
        min_x,min_y=matrice.shape
        max_x,max_y=0,0
        if len(point_set)==1:
            if amplify_point_bbox:
                max_current=0
                max_current_i=[]
                for around in [(-1,1),(1,-1),(-1,-1),(1,1)]:
                    if point_set[0][0]+around[0]<width and point_set[0][0]+around[0]>=0 and point_set[0][1]+around[1]>=0 and point_set[0][1]+around[1]<length:
                        if max_current<=matrice[point_set[0][0]+around[0],point_set[0][1]+around[1]]:
                            max_current_i=[]
                            max_current_i.append((point_set[0][0]+around[0],point_set[0][1]+around[1]))
                min_x, min_y, max_x, max_y=min(max_current_i[0][0],point_set[0][0]),min(max_current_i[0][1],point_set[0][1]),\
                                           max(max_current_i[0][0],point_set[0][0]),max(max_current_i[0][1],point_set[0][1])
            else:
                continue
        else:
            for (x,y) in point_set:
                min_x=min(x,min_x)
                min_y=min(y,min_y)
                max_x=max(x,max_x)
                max_y=max(y,max_y)
        obj_r.append([min_x,min_y,max_x,max_y])
    return obj_r

def map2original_size(obj_outline,current_size,orignal_size):
    '''
    scale the bbox to its original size
    :param obj_outline: output of 'consciousness_matched_detectron'
    :param current_size: feature map size
    :param orignal_size: input size
    :return: scaled obj_outline
    '''
    c_w,c_h=current_size
    o_w,o_h=orignal_size
    scaled_loutline=[]
    for point_set in obj_outline:
        scaled_loutline.append([int(point_set[0]*o_w/c_w),int(point_set[1]*o_h/c_h),int(point_set[2]*o_w/c_w),int(point_set[3]*o_h/c_h)])

    return scaled_loutline


def target_post(image_set,outlines,remove_redundant=True):
    """
    Discribe of Tensor_crop: based on the consciousness model we could gain the prediction of target location or coordinates.
    Then we have to crop the attentioned region from the tensor input, using the input based on the vision model to judge whether
    the consciousness model's prediction is right.

    tensor_mask: outside of outline are would be replaced by zero
    iou_judge:   based on iou and overlap area to predict the plausible region based on two original predict
    remove_redundant_box: for example, if there are current two predictions, and the large area contain the small area, we could
    use the large area to represent them, in another work ,the small area is a redundant box.
    """
    btz,c,w,h=image_set.size()
    def iou_judge(obj1,obj2,threeshold=0.7):

        def are_contact(obj1,obj2):
            outline_=[min(obj2[0],obj1[0]),min(obj2[1],obj1[1]),max(obj1[2],obj2[2]),max(obj1[3],obj2[3])]
            outline_=[min(max(0,outline_[0]),w-1),min(max(0,outline_[1]),h-1),min(max(0,outline_[2]),w-1),min(max(0,outline_[3]),h-1)]
            return outline_

        IOU=iou_compute(obj1,obj2)

        if IOU > threeshold and IOU <=1:
            return True, are_contact(obj1, obj2)
        else:
            return  False, obj1

    def remove_redundant_box(outline):
        for image_id in outline: # img index of the img btz
            specific_pre=outline[image_id]
            specific_pre_update=[]
            for pre in specific_pre:
                specific_outlines=pre[1]
                obj_record=[]
                for specific_outline in specific_outlines:
                    if obj_record==[]:
                        obj_record.append(specific_outline)
                    else:
                        for seq,exist_obj in enumerate(obj_record):
                            op,replace= iou_judge(exist_obj,specific_outline)
                            if op:
                                obj_record[seq]=replace
                            else:
                                obj_record.append(specific_outline)
                specific_pre_update.append((pre[0],obj_record))
            outline.update({image_id:specific_pre_update})
        return outline

    for id_image in range(image_set.size()[0]):
        new_outlines=[]
        for id_image_matched in outlines:
            for id_image_seq in id_image_matched:
                if id_image_seq==id_image:
                    new_outlines.append(id_image_matched)
        outlines=new_outlines
        post_process = {}
        for seqential_pre in range(len(outlines)):
            for target_id in outlines[seqential_pre]:
                if target_id not in post_process:
                    post_process[target_id] = []
                    post_process[target_id].append(outlines[seqential_pre][target_id])
                else:
                    post_process[target_id].append(outlines[seqential_pre][target_id])
    if remove_redundant:
        return  remove_redundant_box(post_process)
    else:
        return post_process

def tensor_crop(imageset,target_predicts):
    """
    :param imageset:
    :param target_predicts:
    :return:
    [
    [ [id1_img1_target1,id1_img1_target2]                ] img1
    ]
    """
    w,h=imageset.size()[2:]
    total_img_record=[]
    for img_id in range(imageset.size()[0]):
        current_image=imageset[img_id]
        for target in target_predicts[img_id]:
            per_img_per_traget=[]
            for outline in target[1]:
                try:
                    resized_crop_tensor=sub_crop_resize(current_image,outline,(w,h))
                except:
                    outline=[min(max(outline[0],0),w-1),min(max(outline[1],0),h-1),min(max(outline[2],0),w-1),min(max(outline[3],0),h-1)]
                    resized_crop_tensor = sub_crop_resize(current_image, outline, (w, h))
                resized_crop_tensor=resized_crop_tensor.unsqueeze(0)
                per_img_per_traget.append((resized_crop_tensor,outline))
        total_img_record.append(per_img_per_traget)
    return total_img_record

def overlap_judge(tuple1, tuple2):
    """
    this method is helpt to judge whether two object has a overlap area, if this overlap area occupy specific percent of
    each kind of object, then contact them up.
    """
    if tuple1[1] >= tuple2[1]:
        if tuple2[1] >= tuple1[0]:
            return True
        else:
            return False
    else:
        if tuple1[1] >= tuple2[0]:
            return True
        else:
            return False
