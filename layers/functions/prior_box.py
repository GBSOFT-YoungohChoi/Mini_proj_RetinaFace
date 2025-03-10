import torch
from itertools import product as product
import numpy as np
from math import ceil


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip'] # 자르기 여부부
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps] #[[80, 80], [40, 40], [20, 20]]
        self.name = "s"

    def forward(self):
        anchors = [] # anchors = [0.00625, 0.00625, 0.05, 0.05] 의 형태로 출력 
        for k, f in enumerate(self.feature_maps): # k=0, f=[80, 80], k=1, f=[40, 40], k=2, f=[20, 20]
            min_sizes = self.min_sizes[k] # self.min_sizes = [[16, 32], [64, 128], [256, 512]]  
            # k=0, min_sizes = [16, 32], k=1, min_sizes = [64, 128], k=2, min_sizes = [256, 512]
            for i, j in product(range(f[0]), range(f[1])):# i=0, j=0, f=[80, 80], i=0, j=1, f=[80, 80], i=1, j=0, f=[80, 80], i=1, j=1, f=[80, 80]
                for min_size in min_sizes:# min_sizes = [16, 32], min_sizes = [64, 128], min_sizes = [256, 512]
                    s_kx = min_size / self.image_size[1] #s_kx = 32/640 = 0.05(앵커의 너비와 높이를 전체 이미지 크기 대비 비율로 나타냄) 
                    s_ky = min_size / self.image_size[0] #s_ky = 32/640 = 0.05
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]] 
                    # j가 0~79의 값을 가질때, x가 0.5~ 79.5의 값을 갖고, 8씩 곱해주면 전체 이미지 사이즈 4픽셀 ~ 636까지의 중심값을 가짐
                    # dense_cx = [0.00625], self.steps=[8, 16, 32], self.image_size=[640, 640]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    # i가 0~79의 값을 가질때, x가 0.5~ 79.5의 값을 갖고, 8씩 곱해주면 전체 이미지 사이즈 4픽셀 ~ 636까지의 중심값을 가짐
                    # dence_cy = [0.00625], self.steps=[8, 16, 32], self.image_size=[640, 640], 0.5=셀의 중심점을 나타내기 위한 오프셋 
                    # 중심값이 dence_cx, dence_cy에 저장됨, 결과값이 비율로 저장됨
                    
                    # cf) dense_cx * image_size[1] = 0.00625 * 640 = x축 4픽셀이 앵커박스의 중심점임
                    # cf) dense_cy * image_size[0] = 0.00625 * 640 = y축 4픽셀이 앵커박스의 중심점임
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
