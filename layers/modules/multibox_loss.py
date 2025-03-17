import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5). 
        # 두 박스의 교집합 영역/합집합 영역 == IoU와 같음 
        # 박스의 교집합 영역이 0.5 이상인 경우에, priorboxes와 ground truth boxes를 매칭시킴
           
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'. 
        # localization target은 ground truth boxes와 priorboxes의 위치적인 오차(offset)를 계산함 -> 동일 오브젝트 안에서 오브젝트 처음부터 주어진 요소나 지점까지의 변위차를 나타내는 정수형
        # 예를들면 'abcdef'를 포함하는 'A' 배열에서 시작지점에서 'c'까지의 거리는 2임 이를 offset이라고 함 
        # RetinaFace나 SSD 모델에서는 Anchor Box와 실제 정답 사이의 offset 값에 variance라는 값을 적용하여 정규화합니다.
        # variance를 쓰는것과 쓰지 않는것을 비교해봐야함 
        # 이거는 검증해봐야할듯 
        
           
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        # Loss_confidence 는 CrossEntropy Loss를 사용하고, Loss_location은 SmoothL1 Loss를 사용함
        # SmoothL1 Loss는 MSELoss보다 이상치에 robust한 손실함수임
        # 몇가지 케이스에서 exploding gradient 문제를 해결하기 위해 사용됨

        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,# loc preds는 위치 예측값, conf preds는 클래스 예측값
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes) # conf_data.shape = torch.size([32, 16800, 2])
                loc shape: torch.size(batch_size,num_priors,4) # loc_data.shape = torch.size([32, 16800, 4])
                # loc_data는 배치크기 32개의 이미지에 대한 16800개의 앵커박스의 x, y, w, h값을 가지고 있음
                priors shape: torch.size(num_priors,4) # priors.shape = torch.size([16800, 4]) == anchor box, input resolution에 따라서 변화함
                # 16800개의 앵커박스의 center point인 cx, cy 와 사이즈인 cw, ch값을 가지고 있음 
                # PriorBox(앵커박스)가 3개의 레이어로 총 16800개의 구성되는 이유 : 12800 + 3200 + 800 = 16800개로 구성 
                # P3 feature map에서 80 * 80 * 2 = 12800 // 한 셀당 생성하는 박스 개수 [16, 32] 2개
                # P4 feature map에서 40 * 40 * 2 = 3200 // 한 셀당 생성하는 박스 개수 [64, 128] 2개
                # P5 feature map에서 20 * 20 * 2 = 800 // 한 셀당 생성하는 박스 개수 [256, 512] 2개
                # RetinaFace의 stride 8, 16, 32 가 각각 적용되어, 640/8 = 80, 640/16 = 40, 640/32 = 20의 앵커박스 생성

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, landm_data = predictions # prediction은 왼쪽 3가지 값이 묶인 튜플형태임임
        # loc_data = [32, 16800, 4], conf_data = [32, 16800, 2], landm_data = [32, 16800, 10] 
        # 여기서 32는 배치사이즈를 의미함, 배치사이즈에 따라 달라짐
        priors = priors # priors = [16800, 4]  == anchor box, input resolution에 따라서 변화함
        """priors = 정해져있음 = 
tensor([[0.0063, 0.0063, 0.0250, 0.0250],
        [0.0063, 0.0063, 0.0500, 0.0500],
        [0.0188, 0.0063, 0.0250, 0.0250],
        ...,
        [0.9250, 0.9750, 0.8000, 0.8000],
        [0.9750, 0.9750, 0.4000, 0.4000],
        [0.9750, 0.9750, 0.8000, 0.8000]], device='cuda:0')"""
        num = loc_data.size(0)  # num = 32
        num_priors = (priors.size(0)) # num_priors = 16800

        # match priors (default boxes) and ground truth boxes (GT값)
        loc_t = torch.Tensor(num, num_priors, 4) # loc_t.shape = torch.Size([32, 16800, 4]) # 각 데이터를 텐서로 만듬
        # loc_t localization값은 cx, cy, w, h 총 4개의 값이 필요함 (GT값)
        landm_t = torch.Tensor(num, num_priors, 10) # landm_t.shape = torch.Size([32, 16800, 10]) # 각 데이터를 텐서로 만듬
        # 랜드마크는 5개의 포인트 총 x, y값 총 10개의 값이 필요함 (GT값)
        conf_t = torch.LongTensor(num, num_priors) # conf_t.shape = torch.Size([32, 16800]) # 각 데이터를 텐서로 만듬
        # long tensor: 텐서의 자료형 변환 == 타입캐스팅 -> int64 텐서 tensor([[0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0], ..., [0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0]])
        for idx in range(num): # num = 32 = batch_size
            truths = targets[idx][:, :4].data # truths = targets[idx][:, :4].data = [num_objs, 4] = [얼굴 개수, 4] = [xmin, ymin, xmax, ymax]
            # idx = 배치에서 몇번째 이미지인지를 나타냄
            # 만약 31번째라면 targets[31]tensor([[0.2490, 0.0332, 0.6709, 0.5996, 0.3430, 0.2734, 0.5308, 0.2309, 0.4493, 0.3691, 0.4103, 0.4719, 0.5521, 0.4329, 1.0000]], device='cuda:0')
            # targets 는 15개의 값으로 구성되어 있음  
            # truths는 GT값의 x_min, y_min, x_max, y_max 값
            labels = targets[idx][:, -1].data # labels = targets[idx][:, -1].data = 1 or -1값을 가짐  1이면 얼굴, -1이면 배경 
            landms = targets[idx][:, 4:14].data # targets = [num_objs, 15] = [얼굴 개수, 15] = [l0_x, l0_y, l1_x, l1_y, l2_x, l2_y, l3_x, l3_y, l4_x, l4_y]
            defaults = priors.data # torch.Size([16800, 4])
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)
            # threshoi
        if GPU:
            loc_t = loc_t.cuda() #loc_t.shape = torch.Size([32, 16800, 4]) == [batch_size, num_priors, 4]
            conf_t = conf_t.cuda()# conf_t.shape = torch.Size([32, 16800]) == [batch_size, num_priors]
            landm_t = landm_t.cuda() # landm_t.shape = torch.Size([32, 16800, 10]) == [batch_size, num_priors, 10]

        zeros = torch.tensor(0).cuda()
        # landm Loss (Smooth L1)
        # Shape: [batch,num_priors,10]
        pos1 = conf_t > zeros # positive1 = conf_t 중에서 0보다 큰값을 찾음
        # pos1 = GT와 
        # pos1.shape = torch.Size([32, 16800])
        num_pos_landm = pos1.long().sum(1, keepdim=True) # 각 배치마다 positive landmark 개수를 저장함 num_pos_landm.shape = torch.Size([32, 1])
        # 앵커박스가 GT와 매칭되었을때 (conf_t > 0)의 개수를 세어서 positive landmark 개수를 저장함
        # num_pos_landm = tensor([[122],[  9],..., [ 58]], device='cuda:0') -> 첫번쨰 이미지에서 positive prior box개수 122개, 두번째 이미지에서 positive prior box개수 9개, ... 32번째 이미지에서 positive prior box개수 58개 라는 의미미
        # num_pos_landm.shape = torch.Size([32, 1])
        N1 = max(num_pos_landm.data.sum().float(), 1) # 배치에 대한 positive prior box의 개수를 모두 더해서 float형으로 변환함
        # num_pos_landm.data.sum() = tensor(1852, device='cuda:0') -> 1852개의 positive prior box가 존재함, 이미지에 따라서 언제든지 바뀔 수 있음 
        # max를 취해주는 이유는 positive prior box개수가 0개가 되어버리면 loss계산에서 0으로 나눠버리는 문제가 발생함으로 이를 해결하기 위해 max()함수를 사용함
        # N1 = tensor(1852., device='cuda:0') 
        # N1은 전체 Prior box의 개수가 되는데, 전체 Positive prior box개수로 나누어 정규화 하는데 활용함 -> scaling factor
        pos_idx1 = pos1.unsqueeze(pos1.dim()).expand_as(landm_data)
        # pos1.unsqueeze(pos1.dim()).shape = torch.Size([32, 16800, 1])
        landm_p = landm_data[pos_idx1].view(-1, 10) # landm_p.shape = torch.Size([1924, 10])
        # landm_data.shape = [32, 16800, 10]
        # pos_idx1.shape = torch.Size([32, 16800, 10]) # True, False로 이루어진 값이 저장됨
        # landm_data[pos_idx1] 를 하게되면 positive prior box에 해당하는 landmark값을 가져옴
        # landm_data[pos_idx1].shape = torch.Size([19240]) -> 해당 값을 10개씩 나누어주게게되면 [1924, 10]의 형태로 변환됨
        # landm_p.shape = torch.Size([1924, 10])
        # landm_p = landm_data[pos_idx1].view(-1, 10) = [num_pos_landm, 10] = [1852, 10], 10은 랜드마크 포인트임 
        landm_t = landm_t[pos_idx1].view(-1, 10)
        # landm_t.shape = torch.Size([1924, 10])
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        # smmoth l1 loss -> L2의 장점과 L1의 장점을 합쳐, 특정 임계값보다 낮을때 L2loss, 임계값보다 클 때 L1 loss처럼 동작하게 됨 
        # loss_landm = tensor(34456.5625, device='cuda:0', grad_fn=<SmoothL1LossBackward0>), 각각의 loss를 sum으로 모두 더해줌
        pos = conf_t != zeros # zeros = tensor(0, device='cuda:0')
        # pos.shape = torch.Size([32, 16800]) -> conf_t가 0이 아닌 값을 모두 가져옴 
        # pos는 16800개 중에서 
        conf_t[pos] = 1
        # conf_t = tensor([[0, 0, 0,  ..., 0, 0, 0], [0, 0, 0,  ..., 0, 0, 0],...,[0, 0, 0,  ..., 0, 0, 0],[0, 0, 0,  ..., 0, 0, 0]], device='cuda:0')
        # conf_t.shape = torch.Size([32, 16800]), 0 or 1로 이루어져있음음
        # conf_t[pos] 하는 이유는 GT와 매칭된 앵커박스를 1로 설정하는 것 
        # conf_t에서 1이 포함된 값만 연산하기 위해 
        # conf_t[pos].shape = torch.Size([1858])
        # 16800개 중에서 1858개만 가져옴  -> pos가 True인 값이 1858개였다는 뜻 임 
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # pos.shape = torch.Size([32, 16800])
        # pos.dim() = 2
        # pos.unsqueeze(pos.dim()).shape = torch.Size([32, 16800, 1]) -> 2번 인덱스를 새로운 차원으로 느리겠다 
        loc_p = loc_data[pos_idx].view(-1, 4) 
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) 
       

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
