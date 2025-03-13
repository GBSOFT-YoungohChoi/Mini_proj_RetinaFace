import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH



class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        # 기존: (32, 12, 80, 80) → CNN의 일반적인 형식
        # 변경: (32, 80, 80, 12) → 바운딩 박스 좌표를 쉽게 다룰 수 있도록 변경


        return out.view(out.shape[0], -1, 10)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            # mobilenet0.25가 있을 때, MobileNetV1 백본 가져옴 
            if cfg['pretrain']:
                checkpoint = torch.load("./weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                # pretrain = True일때, pretrain.tar을 가져옴
                from collections import OrderedDict # layer와 weight를 순서대로 저장학 위해 ordereddict를 불러옴 
                # 일반적인 dict를 사용하게 되면 key값(layer이름) 매칭이 정확히 되지 않아 - prefix(layer이름 앞에 붙는 접두사 - "module."), weights들이 정해진 layer로 들어갈 수 없음
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove "module."
                    new_state_dict[name] = v
                    # 각 layer의 이름과 weight값들을 저장하게됨 
                    # v.shape = torch.Size([1000]) # fc.bias 레이어의 weights의 shape찍어봄
                    # new_state_dict = OrderedDict([('stage1.0.0.weight', tensor([[[[ 0.0516,  0.0249, -0.0566],...('stage1.5.1.running_var', tensor([0.0185, 0.0259, 0.0489, 0.0338, 0.0116, 0.0025, 0.0873, 0.0483, 0.0134, ..., 0.0518, 0.0184, 0.0175, 0.0816])), ...])) [-0.3015,  0.2490,  0.4354],
                    # new_state_dict 는 총 164 이지만 164개의 layer라고 판단할 수 없음 -> weight와 bias가 같이 있으므로, ordered dict에 weight뿐아니라bias와 variance도 포함되어 있음 
                # # Load params
                backbone.load_state_dict(new_state_dict) 
                # backbone MobileNet에 각 레이어에 맞는 weight를 가져옴 
                 
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        # 모델의 중간 레이어의 출력을 추출하는 역할을 하는 nn.ModuleDict 기반의 래퍼 클래스
        # 순차적으로 등록된 레이어만 추출 가능
        # 직접 할당된 서브모듈만 접근 가능 # model.feature1 같은 1단계 서브모듈은 접근 가능하나, model.feature1.layer2 처럼 2단계 이상 들어가면 접근 불가능
        # 여기서 헤드를 빼주게됨
        
        in_channels_stage2 = cfg['in_channel']
        # stage2의 inchannel에 cfg에 있는 "in_channl" : 32가 들어가게됨 
        # 
        in_channels_list = [
            in_channels_stage2 * 2, # 64
            in_channels_stage2 * 4, # 128
            in_channels_stage2 * 8, # 256
        ]
        out_channels = cfg['out_channel'] # out_channels에 ['out_channel': 64]가 할당됨
        self.fpn = FPN(in_channels_list,out_channels) # FPN이란 Feature Pyramid Network를 의미함
        # self.fpn = [output1, output2, output3] == [output1+output2, output2+output3, output3] 인것임
        # output1 = output1 + output2 (output1이 고해상도 feature map임)
        # output2 = output2 + output3
        # output3 = output3
        self.ssh1 = SSH(out_channels, out_channels) # self.ssh1 = Relu(3x3와 5x5와 7x7의  receptive field를 모두 concat한 값)
        self.ssh2 = SSH(out_channels, out_channels) # self.ssh2 = Relu(3x3와 5x5와 7x7의  receptive field를 모두 concat한 값)
        self.ssh3 = SSH(out_channels, out_channels) # self.ssh3 = Relu(3x3와 5x5와 7x7의  receptive field를 모두 concat한 값)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self,inputs):
        out = self.body(inputs)
        # inputs.shape = torch.Size([32, 3, 640, 640]) 640*640크기의 이미지 32개를 모델에 넣음음 
        # for key, value in out.items():    print(f"Stage {key}: {value.shape}") 
        # Stage 1: torch.Size([32, 64, 80, 80]) # MobileNet의 1stage에서 출력된 텐서
        # Stage 2: torch.Size([32, 128, 40, 40]) # MobileNet의 2stage에서 출력된 텐서
        # Stage 3: torch.Size([32, 256, 20, 20]) # MobileNet의 3stage에서 출력된 텐서
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0]) # feature1.shape = torch.Size([32, 64, 80, 80])
        feature2 = self.ssh2(fpn[1]) # feature2.shape = torch.Size([32, 64, 40, 40])
        feature3 = self.ssh3(fpn[2]) # feature3.shape = torch.Size([32, 64, 20, 20])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        # bbox_regressions.shape = torch.Size([32, 16800, 4]) 
        # 16800개의 앵커박스에 대한 Bbox좌표 예측
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        # classifications.shape = torch.Size([32, 16800, 2])
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output