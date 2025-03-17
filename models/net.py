import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module): # Single stage Headless face detector 
    """Fully connected network없이 오롯이 convolution layer만 사용
    # Fully connected network 없이도 CNN만으로도 예측이 가능하다는것이 핵심
    # structure 특징 : 
    # 1. FCN을 제거하여 가벼운 Network
    # 2. Scale불변성 :외부 Mulitiscale pyramid 없이도 다양한 Scale의 얼굴을 검출
    # 3. 서로 다른 stride를 가진 convolution module을 각 layer에 배치함.
    # 4. 속도가 ResNet대비 5배빠름 
    # 5. Performance 상승"""
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)
        # Conv2d + BatchNorm (relu없음)
        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        # conv_bn = Conv2d + BatchNorm + LeakyRelu
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        # 실제로는 3x3 필터 conv가 2번 적용되서 5x5와 같은 receptive field를 가지게 된다는 의미임
        conv5X5 = self.conv5X5_2(conv5X5_1)
        # conv5x5는 적용이 안됨 
        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        # receptive field의 크기가  3x3을 3번 적용한것과 같음
        conv7X7 = self.conv7x7_3(conv7X7_2) 

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1) 
        # 3x3와 5x5와 7x7의  receptive field를 모두 concat하게됨 
        out = F.relu(out) # 렐루로 마무리 
        return out 
        # out.shape = torch.Size([32, 64, 40, 40])
        
class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        # conv2d(filter 1을 적용) + bn, in_channels_list[0] = 64
        # 1x1을 통해 채널간 정보 선형 결합하여 압축
        # 채널수를 64로 출력함
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        # conv2d(filter 1을 적용) + bn, in_channels_list[1] = 128 
        # 1x1을 통해 채널간 정보 선형 결합하여 압축
        # 채널수를 64로 출력함
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)
        # conv2d(filter 1을 적용) + bn, in_channels_list[2] = 128
        # 1x1을 통해 채널간 정보 선형 결합하여 압축
        # 채널수를 64로 출력함
        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values()) # [64, 128, 256]

        output1 = self.output1(input[0]) # conv_bn1X1를 통한 채널 선형결합 input = 64, out_channels=64
        output2 = self.output2(input[1]) # conv_bn1X1를 통한 채널 선형결합 input = 128, out_channels=64
        output3 = self.output3(input[2]) # conv_bn1X1를 통한 채널 선형결합 input = 256,  out_channels=64
        # 세가지 모두 채널수를 64로 출력하여 결합이 가능하도로고 만듦듦
        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        # feature map을 결합하기 위해 (feature pyramid network)
        # 서로 다른 output3와 output2를 결합하기 위해서 상위 계층의 feature map인 output3를 하위계층 feature map인 output2의 크기로 업샘플링함 
        # 고압축된 feature map 을 중압축된 feature map의 사이즈로 맞추는 과정인거지 
        output2 = output2 + up3 # 두 feature를 결합함
        output2 = self.merge2(output2) # bn 을 통해 feature 정제 

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        # output2를 최근접이웃근사를 통해 output1 feature map과 동일한 사이즈를 만들어줌
        output1 = output1 + up2 
        #두 피처맵을 합쳐줌 
        output1 = self.merge1(output1)
        # bn을 통해 featrue 정제 
        out = [output1, output2, output3]
        # 세 output feature를 결합해줌 
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

