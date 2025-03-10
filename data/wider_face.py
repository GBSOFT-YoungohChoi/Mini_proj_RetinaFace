import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape # 이미지의 height = 754, width = 1024, _=3 (RGB), img.shape = (754, 1024, 3)

        labels = self.words[index] # labels = [[40.0, 2.0, 126.0, ..., 108.0, 1.0, ...]* 얼굴 n개]의 형태로 저장됨
        annotations = np.zeros((0, 15)) # annotations = [] 빈 배열로 저장됨 -> shape = (0, 15) 
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):# 레이블분리 및 넘버링
            annotation = np.zeros((1, 15)) # array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]) -> shape = (1, 15)
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]    # l0_x
            annotation[0, 5] = label[5]    # l0_y
            annotation[0, 6] = label[7]    # l1_x
            annotation[0, 7] = label[8]    # l1_y
            annotation[0, 8] = label[10]   # l2_x
            annotation[0, 9] = label[11]   # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y
            annotation[0, 12] = label[16]  # l4_x
            annotation[0, 13] = label[17]  # l4_y 
            # annotation = array([[ 40. ,   2. , 166. , 160. ,  82.786,  54. , 140.214, 60. , 117.071,  85.714,  99.929, 107.143, 128.214, 108. , 0.   ]]) -> shape = (1, 15)로 구성됨
            if (annotation[0, 4]<0): # 왼쪽눈의 랜드마크 좌표를 기준으로 랜드마크 가시성 정보 annotation[0, 14] 저장
                annotation[0, 14] = -1 
            else:
                annotation[0, 14] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations) # target = annotations array로 변환
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):#batch = [img, target]*32(batch_size) 의 형태로 저장됨
        # batch[0][0].shape = torch.Size([3, 640, 640]) -> 이미지의 shape = (3, 640, 640)
        # batch[0][1].shape = torch.Size([2, 15]) -> 어노테이션의 shape = (2, 15) -> 한사진안에 얼굴 개수가 2개인거임 
        for _, tup in enumerate(sample):
            # sample[0].shape = torch.Size([3, 640, 640]) -> 이미지의 shape = (3, 640, 640)
            # sample[1].shape = torch.Size([2, 15]) -> 어노테이션의 shape = (2, 15) -> 한사진안에 얼굴 개수가 2개인거임
            # sample = (img, target), type: tuple
            # tup은 img or annotation 각각으로 분리됨
            # tup = array([[0.3040293 , 0.04395604, 0.63003663, 0.51648352, 0.47295238,0.24174237, 0.60910501, 0.2535812 , 0.58246642, 0.32461783, 0.46407326, 0.3808547 , 0.58542613, 0.40749328, 1.        ]])
            # tup.shape = (1, 15) or (3, 640, 640)
            if torch.is_tensor(tup): 
                imgs.append(tup) # img는 tensor임으로, 변환없이 바로 이미지에 추가됨 
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float() # annotation은 numpy이므로, isinstance로 numpy배열인지체크하고, tensor로 변환하여 추가함
                targets.append(annos)

    return (torch.stack(imgs, 0), targets) # 텐서형태의 이미지와 텐서형태의 annotation 을 concat해줌  
    # return (tensor([[[[ -26.,  -19.,  -11.,  ...,  -88.,  -88.,  -88.], [ -33.,  -26., ...61.,  -64.,  ...,  -38.,  -38.,  -44.]]]]), [tensor([[0.0265, 0.1106, 0.3584, 0.4912, 0.1271, 0.2546, 0.2794, 0.2522, 0.2057,    ... 0.6688, 0.5846, 0.8032, 0.5926, 1.0000]]), tensor([[0.3040, 0.0440, 0.6300, 0.5165, 0.4730, 0.2417, 0.6091, 0.2536, 0.5825,    ... 0.4641, 0.3809, 0.5854, 0.4075, 1.0000]]), tensor([[0.0905, 0.2421, 0.1760, 0.3741, 0.1084, 0.2942, 0.1185, 0.2900, 0.0967,...
    # return shape (3, 640, 640), (1, 15) or (3, 640, 640), (2, 15) or (3, 640, 640), (3, 15)의 형태로 반환됨