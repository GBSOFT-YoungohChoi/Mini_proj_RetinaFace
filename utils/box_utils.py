import torch
import numpy as np


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
        
    # 현재의 boxes는 중심좌표 기반으로 되어 있음 boxes = (cx, cy, w, h)
    # 좌상단과 우하단 코너좌표로 변환하기 위해서는 cx 에서 w/2, cy에서 h/2를 빼고 더해주면 됨 
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    # box_a = GT값
    # box_a.shape = torch.Size([1, 4])
    # box_b = 앵커박스 값
    # box = [좌상단 x, 좌상단 y, 우하단 x, 우하단 y] 값으로 존재함 
    # # (cx, cy, w, h)에서 좌상단 좌하단값으로 변화하는 함수인 point_form을 통해 변환됨
    # box_b.shape = torch.Size([16800, 4])
    """
    A = box_a.size(0) # 박스 A의 개수 # GT에서 얼굴이 1개면 A = 1
    B = box_b.size(0) # 박스 B의 개수 # 앵커박스의 개수 == 16800
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), # box[:, 2:]는 박스의 우측하단 좌표 (x2, y2) 의미, unsqueeze를 통해 차원을 늘림 
                       # box_a[:, 2:] = tensor([[0.4453, 0.2051]], device='cuda:0')
                       # box_a[:, 2:].unsqueeze(1) = tensor([[[0.4453, 0.2051]]], device='cuda:0')
                       # box_a[:, 2:].unsqueeze(1).expand(A, B, 2).shape = torch.Size([1, 16800, 2]) -> GT값의 우하단 좌표값을 3차원으로 변환함
                       # unsqueeze(1)을 통해 차원을 늘림 뒤쪽에 차원을 늘려줌 [3]->[3, 1]
                       # unsqueeze(0)을 통해 차원을 늘림 앞쪽에 차원을 늘려줌 [3]->[1, 3]
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2)) # box_b[:, 2:].unsqueeze(0).expand(A, B, 2).shape = torch.Size([1, 16800, 2]) -> 앵커박스의 우하단 좌표값을 3차원으로 변환함
                       # box_b[:, 2:] = tensor([[0.0188, 0.0188], [0.0312, 0.0312],[0.0312, 0.0188],...,[1.3250, 1.3750],[1.1750, 1.1750],[1.3750, 1.3750]], device='cuda:0')
                       # box_b[:, 2:].shape = torch.Size([16800, 2])
                       # box_b[:, 2:].unsqueeze(0).shape = torch.Size([1, 16800, 2])
                       # box_b[:, 2:].unsqueeze(0).expand(A, B, 2).shape = torch.Size([1, 16800, 2])
                       # 단순히 unqueeze(0)를 하게되면 A = 5개 일 때, GT박스와의 비교가 어려움움
                        # expand(A, B, 2)를 통해 브로드캐스팅방식으로 차원을 늘림 
                        # max 값들(우하단좌표) 중에서 가장 작은 값을 찾아야 박스가 겹치는 부분을 구할 수 있음 
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))  #겹치는 부분을 찾기위해 min중에 최대값을 찾고, max중에 최소값을 찾아야하므로 
                    # 구하는 방식은 max_xy와 같음
                    # min 값들(좌상단 좌표) 중에서 가장 큰 값을 찾아야 박스가 겹치는 부분을 구할 수 있음
    inter = torch.clamp((max_xy - min_xy), min=0) # GT박스와 앵커박스의 교집합 박스의 너비와 높이를 계산하고, 교집합이 없을 경우 음수가 되지 않도록 min=0으로 설정함 
    # inter.shape  = torch.Size([1, 16800, 2])
    return inter[:, :, 0] * inter[:, :, 1] # GT박스와 앵커박스의 교집합 박스의 너비와 높이로 면적을 구함 


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. 
    # IoU와 똑같다는 얘기 
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4] # 얼굴의 GT값 [얼굴개수,bounding boxes gt값 4개]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4] # 얼굴의 예측 앵커박스 값 
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]

    # box_a = GT값
    # box_a.shape = torch.Size([1, 4])
    # box_b = 앵커박스 값
    # box = [좌상단 x, 좌상단 y, 우하단 x, 우하단 y] 값으로 존재함 
    """
    inter = intersect(box_a, box_b) 
    # intersect 함수의 return값은 "inter[:, :, 0] * inter[:, :, 1]"값으로, GT박스와 앵커박스의 교집합 박스의 너비와 높이로 면적을 구함 
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # box_a
    # area_a.shape = (18,)
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)


def matrix_iof(a, b):
    """
    return iof of a and b, numpy version for data augenmentation
    # 데이터 증강을 위한 numpy 버전의 iof를 반환함
    IOF: A∩𝐵/area(A)​
  → GT 박스(또는 특정 박스)의 영역을 기준으로 IoU 계산

    a = boxes = [x_min, y_min, x_max, y_max]
    b = crop ROI (multi scale로 원본이미지를 자를 범위에 대한 x1, y1, x2, y2)
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2]) # 좌상단 좌표 비교 (left top)
    # a[0] = array([158.,  70., 332., 300.])
    # a.shape = (2, 4) -> GT에 따라서 계속 바뀜 
    # b = array([[196,   0, 967, 771]])
    # b.shape = (1, 4)
    # a[:, np.newaxis, :2]로 np.newaxis를 하게되면 브로드캐스팅으로 인해 shape이 (2, 1, 4)가 되어, b의 shape인 (1, 4)와 겹치는 부분을 계산할 수 있게 됨 
    # np.newaxis를 하지 않으면 계산불가
    # lt.shape = (얼굴개수 , 1, 2)
    # 겹치는 부분을 구하기 위해 좌상단 좌표를 비교함 x_min, y_min중에서 maximum값을 쓰면 겹치는 부분을 구할 수 있음
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:]) # 우하단 좌표 비교(right bottom)
    # rb.shape = (18, 1, 2)
    # 교집합 영역 계산
    # a[:, np.newaxis, 2:] = array([[[x_min, y_min]], [[x_min, y_min]]])
    # 겹치는 부분을 구하기 위해 좌상단 좌표를 비교함 x_max, y_max중에서 maximum값을 쓰면 겹치는 부분을 구할 수 있음
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2) # prod = production 임 --> 가로 길이 * 세로길이 = 박스면적(교집합 영역) 
    # a 박스의 영역 (고정된 기준 박스)
    # (lt < rb).all(axis=2) -> 교집합이 존재하는 경우만 남김, 겹치는 부분이 없으면 0
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    # a[:, 2:] = [x_max, y_max] = boxes의 우하단
    # a[:, :2] = [x_min, y_min] = boxes의 좌상단을 의미하므로 
    # np.prod를 통해 w, h값을 서로 곱해주어 겹치는 면적을 구함
    # IOF 계산 (교집합 / a 박스의 면적)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    # jaccard overlap(IoU =(A ∩ B) / (A ∪ B))이 가장 높은 ground truth box와 각 prior box를 일치시키고, bounding box를 인코딩한 다음 일치하는 인덱스를 반환하십시오
    Args:
        threshold: (float) The overlap threshold used when mathing boxes. # IoU 임계값
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4]. # x1, y1, w, h에 대한 GT값 -> 실제 얼굴 위치 정보 
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4]. # 앵커박스의 cx, cy, w, h값
        variances: (tensor) Variances corresponding to each prior coord, # 정답박스와 예측된 앵커박스간의 위치차이를 나타내는 offset을 정규화할 때 사용하는 값
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj]. # 각 얼굴에 대한 클래스 레이블, 얼굴 = 1, 배경 = 0
        landms: (tensor) Ground truth landms, Shape [num_obj, 10]. # 랜드마크 GT값, 왼쪽 눈, 오른쪽 눈, 코, 왼쪽 입 끝 위치, 오른쪽 입 끝 위치
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets. # localization target == 정답역할을 할 텐서 
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds. # confidence target == 클래스 예측값을 저장할 텐서
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets. # landm target == 랜드마크 예측값을 저장할 텐서
        idx: (int) current batch index # 현재 배치 인덱스
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds. # 위치, confidence, 랜드마크
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # overlaps = [2, 16800]
    # truths = tensor([[0.0000, 0.1768, 0.2478, 0.7868],[0.4835, 0.1837, 0.9099, 0.7730]], device='cuda:0')
    # truths.shape = torch.Size([2, 4])
    # point_form(priors) =  torch.Size([16800, 4]) 
    # point_form 함수 -> 앵커박스의 센터값이 중심이었던 값을 (xmin, ymin, xmax, ymax)로 변환함
    # 두 박스의 IoU값을 계산함 
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    matches_landm = landms[best_truth_idx]
    landm = encode_landm(matches_landm, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    landm_t[idx] = landm


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4]. 
            # [GT 개수, 좌상단 x, 좌상단 y, 우하단 x, 우하단 y]
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
            # [앵커박스 개수, cx, cy, w, h]
        variances: (list[float]) Variances of priorboxes
        # variances = [0.1, 0.2]
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # matched.shape = torch.Size([16800, 5, 2])
    # matched는 GT박스의 좌상단, 우하단 좌표값을 가지고 있음
    # matched[:, :2].shape = torch.Size([16800, 2, 2])
    # matched 의 의미 -> 16800개의 앵커박스 각각이 5개의 GT박스를 저장하도록 설계함 -> 여러 얼굴을 감지하는 성능이 향상됨 
    # matched[prior_idx][gt_idx] = [x_min, y_min]
    # priors[:, :2].shape = torch.Size([16800, 2, 4])
    # g_cxcy 는 GT박스의 센터좌표를 나타냄
    
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # priors[:, 2:].shape = torch.Size([16800, 3, 4])
    # variances[0] = 0.1
    
    # match wh / prior wh
    # matched.shape  = torch.Size([16800, 4])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:] # matched = 우하단에서 좌상단값을 빼주어 GT값의 w, h값을 구해줌
    # matched[:, 2:] = GT값의 x_max, y_max 우하단 위치 값
    # matched[:, :2] = GT값의 x_min, y_min 좌상단 위치 값
    # (matched[:, 2:] - matched[:, :2]) = GT값의 w, h값
    # priors[:, 2:] = 앵커박스의 w, h값
    # g_wh = 각각의 GT박스의 크기가 앵커박스의 크기대비 얼마나 큰지를 나타내는 비율 값 
    # 또한 앵커박스의 크기가 고정된 크기로 정의되므로, GT박스의 크기를 앵커박스의 크기로 정규화하여 학습을 안정화시킬 수 있음
    
    # 이후 prior의 w, h값으로 나눠주어, priors의 
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # 두 텐서를 concat해줌
    # g_cxcy.shape = torch.Size([16800, 2])
    # g_wh.shape = torch.Size([16800, 2])
    # torch.cat([g_cxcy, g_wh], 1) = torch.Size([16800, 4])

def encode_landm(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded landm (tensor), Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    # g_cxcy /= priors[:, :, 2:]
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    # return target for smooth_l1_loss
    return g_cxcy


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    # 학습 시 수행한 offset regression을 원래 좌표로 변환하는 함수
    # 즉, 모델이 예측한 위치(loc)를 실제 이미지 좌표로 디코딩하는 과정
    Args:
        loc (tensor): location predictions for loc layers,
        # 모델이 예측한 박스 오프셋 (offset regression 결과).
            Shape: [num_priors,4](x_offset, y_offset, w_offset, h_offset)
        priors (tensor): Prior boxes in center-offset form. # 미리 정의된 anchor box (Prior Boxes)
            Shape: [num_priors,4].(x_center, y_center, width, height)
        variances: (list[float]) Variances of priorboxes
        # Prior boxes에서 사용한 변동성 값 (scale factor)
        # variances = [0.1, 0.2]
    Return:
        decoded bounding box predictions
        디코딩된 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # priors[:, :2].shape = torch.Size([23936, 2]) # 기준이 되는 anchor box의 중심 좌표 (x_center, y_center)  (shape: [num_priors, 2])
    # loc[:, :2].shape = torch.Size([23936, 2]) # 모델이 예측한 x, y 중심 좌표의 오프셋 (shape: [num_priors, 2])
    # priors[:, 2:]: anchor box의 width, height (shape: [num_priors, 2])
    # variances = [0.1, 0.2]  scale factor
    # priors.shape = torch.Size([23936, 4])
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(scores.size(0)).fill_(0).long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count


