"""
WiderFace evaluation code
author: wondervictor
mail: tianhengcheng@gmail.com
copyright@wondervictor
"""

import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
from bbox import bbox_overlaps
from IPython import embed


def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat')) # type(gt_mat) = <class 'dict'>
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    # gt_mat['face_bbx_list'].shape = (61, 1)
    # bbox에 대한 정보 
    event_list = gt_mat['event_list']
    # event_list = [[array(['0--Parade'], dtype='<U9')] [array(['1--Handshaking'], dtype='<U14')]]
    # 파일 이름 
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list'] # hard_mat['gt_list'].shape = (61, 1)
    medium_gt_list = medium_mat['gt_list'] # medium_mat['gt_list'].shape = (61, 1)
    easy_gt_list = easy_mat['gt_list'] # easy_mat['gt_list'].shape = (61, 1)

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list


def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes


def read_pred_file(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()
        img_file = lines[0].rstrip('\n\r')
        lines = lines[2:]

    # b = lines[0].rstrip('\r\n').split(' ')[:-1]
    # c = float(b)
    # a = map(lambda x: [[float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4])] for a in x.rstrip('\r\n').split(' ')], lines)
    boxes = []
    for line in lines:
        line = line.rstrip('\r\n').split(' ')
        if line[0] is '':
            continue
        # a = float(line[4])
        boxes.append([float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])
    boxes = np.array(boxes)
    # boxes = np.array(list(map(lambda x: [float(a) for a in x.rstrip('\r\n').split(' ')], lines))).astype('float')
    return img_file.split('/')[-1], boxes


def get_preds(pred_dir):
    events = os.listdir(pred_dir)
    boxes = dict()
    pbar = tqdm.tqdm(events)
    
    for event in pbar:
        pbar.set_description('Reading Predictions ')
        event_dir = os.path.join(pred_dir, event)
        #event_dir = /home/youngoh/retinaface/Pytorch_Retinaface/widerface_evaluate/widerface_txt/13--Interview'
        # dir 위치랑 파일 이름 조인 
        event_images = os.listdir(event_dir)
        current_event = dict()
        for imgtxt in event_images:
            # len(event_images) = 142
            # event_images = ['13_Interview_Interview_Sequences_13_1032.txt',...,]
            imgname, _boxes = read_pred_file(os.path.join(event_dir, imgtxt))
            # imgname ='24_Soldier_Firing_Soldier_Firing_24_812'
            # _boxes.shape = (7, 5) # _boxes = [x_min, y_min, x_max, y_max]값임 
            current_event[imgname.rstrip('.jpg')] = _boxes
            #imgname.rstrip('.jpg') = '24_Soldier_Firing_Soldier_Firing_24_812' #.jpg제거
        boxes[event] = current_event
    return boxes


def norm_score(pred):
    """ norm score
    pred {key: [[x1,y1,x2,y2,s]]}
    pred = {key: [[Bbox x1,Bbox y1,Bbox x2,Bbox y2,confidence score]]}를를 의미함
    """

    max_score = 0
    min_score = 1

    for _, k in pred.items():
        # pred = {'13--Interview': {'13_Interview_Interview_Sequences_13_1032': array([[ 3.9300000e+02,  1.4100000e+02,  2.0300000e+02,  2.7500000e+02, 9.99...  2.7600000e+02, 2.0409448e-02]]),...
        # k = {'13_Interview_Interview_Sequences_13_1032': array([[ 3.9300000e+02,  1.4100000e+02,  2.0300000e+02,  2.7500000e+02, 9.99...  2.7600000e+02,
        # len(pred) = 61
        for _, v in k.items():
            if len(v) == 0: # len(v) = 20 이면 skip안됨 -> 예측값이 있는지 없는지 확인하는 함수, 예측된 바운딩박스가 없으면 무시됨됨 
                continue
            _min = np.min(v[:, -1]) # 현재 이미지에서 가장 낮은 confidence score
            _max = np.max(v[:, -1]) # 현재 이미지에서 가장 높은 confidence score max
            max_score = max(_max, max_score) # _max = 0.98726374 -> 전체에서 가장높은 confience score 찾기 
            min_score = min(_min, min_score)# _min = 0.020323357 -> 전체에서 가장낮은은 confience score 찾기 

    diff = max_score - min_score # max에서 min값을 빼줌 # 이값을 사용해서 점수를 0~1의 범위로 변환함
    for _, k in pred.items():
    
        for _, v in k.items():
            if len(v) == 0: #len(v) = 15,v = array([[ 3.9300000e+02,  1.4100000e+02,  2.0300000e+02,  2.7500000e+02, ...] 9.9987495e-01],
                # v = [x_min, y_min, x_max, y_max, confidence score]
                continue
            v[:, -1] = (v[:, -1] - min_score)/diff # min, max 정규화 공식을 사용하여 변환해줌
            # min-max정규화를 수행하면 다른 데이터셋과 비교하기 쉬워짐 
            # 일관된 기준으로 평가할 수 있음 
            # precision-recall curve(PR Curve)를 계산할 때 score가 0~1범위에 있어야 제대로 된 곡선을 그릴 수 있음  


def image_eval(pred, gt, ignore, iou_thresh):
    """ single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """

    _pred = pred.copy() # _pred.shape =  (794, 5) = 모델이 예측한 바운딩 박스와 confidence score
    _gt = gt.copy() # _gt.shape =(126, 4) # Ground Truth 바운딩박스 좌표
    pred_recall = np.zeros(_pred.shape[0])
    #pred_recall.shape = (794,)
    recall_list = np.zeros(_gt.shape[0])
    # recall_list.shape = (126,)
    proposal_list = np.ones(_pred.shape[0])
    # proposal_list.shape = (794,)
    # pred_recall, recall_list, proposal_list 초기화
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    # x2 = x1 + width
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    # y2 = y1 + height
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    # x2 = x1 + width (GT)
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]
    # y2 = y1 + height (GT)
    overlaps = bbox_overlaps(_pred[:, :4], _gt)
    # 박스간의 IoU 계산 
    for h in range(_pred.shape[0]):
    # 모든 예측된 바운딩 박스를 순회하면서 가장 높은 IoU 값을 찾기 위한 반복문
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        # 가장 높은 IoU 값을 가진 GT 찾기
        # gt_overlap.max() → 가장 높은 IoU 값 (max_overlap)
        # gt_overlap.argmax() → IoU 값이 가장 높은 GT 박스의 인덱스 (max_idx)
        if max_overlap >= iou_thresh:
        #  IoU가 iou_thresh 이상이면, 예측된 박스가 GT와 매칭되었다고 판단
            if ignore[max_idx] == 0: # GT가 무시되면 recall 및 proposal 값을 -1로 설정
                recall_list[max_idx] = -1 
                proposal_list[h] = -1 # proposal_list[h] = -1 → 해당 예측된 박스도 유효하지 않다고 간주
            elif recall_list[max_idx] == 0: # recall_list[max_idx] == 0 → 해당 GT가 아직 매칭되지 않았을 경우
                recall_list[max_idx] = 1 # recall_list[max_idx] = 1 → 예측된 박스가 해당 GT를 찾았다고 표시

        r_keep_index = np.where(recall_list == 1)[0] # np.where(recall_list == 1)[0] → Recall이 1인 GT 박스들의 인덱스만 추출
        pred_recall[h] = len(r_keep_index) # pred_recall[h] → 예측된 박스 h번째까지 Recall이 발생한 GT 개수 저장
    return pred_recall, proposal_list 
# pred_recall: 각 예측된 박스에 대해 recall 값 저장
# proposal_list: 최종적으로 유효한 예측 박스를 나타내는 리스트 (1이면 유효, -1이면 무효)
def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    # thresh_num: 신뢰도(Confidence Score) 임계값 개수 (보통 1000개)
    # pred_info: 예측된 박스 정보 (Nx5) # pred_info[:, 4] → Confidence Score 리스트
    # proposal_list: Proposal 여부 (Nx1) # 값이 1이면 Proposal로 유지, -1이면 제거됨.
    # pred_recall: Recall 값 리스트
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    # Precision-Recall 저장을 위한 배열 생성
    for t in range(thresh_num): # 1000개(thresh_num) threshold 값에 대해 반복문 수행

        thresh = 1 - (t+1)/thresh_num
        # Confidence Score 임계값(threshold) 계산
        # t=0 → thresh = 0.999
        # t=1 → thresh = 0.998
        # t=999 → thresh = 0.000
        # 즉, Confidence Score가 높은 것부터 낮은 것까지 반복하며 Precision-Recall을 계산함.
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        # 현재 threshold(Confidence Score) 이상인 박스들의 인덱스 찾기
        # pred_info[:, 4] → Confidence Score 리스트
        # r_index → Confidence Score ≥ thresh인 박스들의 인덱스 리스트
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        # 해당 Confidence Score 이상인 박스가 없으면 Precision & Recall = 0
        # r_index가 비어 있다면 Precision과 Recall도 0으로 설정

        else:
            r_index = r_index[-1]
            # 가장 마지막 인덱스(최대 인덱스) 선택
            # r_index[-1] → r_index 리스트에서 가장 마지막 요소 (가장 작은 Confidence Score)
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            # Proposal(1인 값) 인덱스 찾기
            # proposal_list[:r_index+1] == 1 → Proposal로 유지된 박스만 선택
            pr_info[t, 0] = len(p_index)
            # Precision: 유지된 Proposal 개수 저장
            # len(p_index) → 유지된 Proposal 개수를 pr_info[t, 0]에 저장
            pr_info[t, 1] = pred_recall[r_index]
            # Recall: r_index에서의 Recall 값을 저장
            # pr_info[t, 1] = pred_recall[r_index]
            # 해당 Confidence Score에서의 Recall 값을 저장
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluation(pred, gt_path, iou_thresh=0.5):
    # pred: 모델이 예측한 바운딩 박스 파일이 저장된 디렉터리 (예: ./widerface_txt/)
    # gt_path: Ground Truth 파일이 저장된 디렉터리 (예: ./ground_truth/)
    # iou_thresh=0.5: IoU(Intersection over Union) 임계값 (기본적으로 0.5)
    pred = get_preds(pred)
    # get_preds(pred): pred 디렉터리에 저장된 모델 예측값(.txt 파일) 을 Dictionary 형태로 변환
    # 예측 진행행
    norm_score(pred)
    # 모든 예측 박스의 Confidence Score를 [0,1] 범위로 정규화
    # score 값을 min-max 정규화하여 학습할 때 분포를 맞춰주는 과정
    facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list = get_gt_boxes(gt_path)
    # facebox_list.shape = (61, 1) # 바운딩박스 4개좌표  # x, y, w, h임 좌표로 찍혀있음, 
    # event_list.shape = (61, 1) # 카테고리
    # file_list.shape = (61, 1) # 파일이름
    # hard_gt_list.shape = (61, 1) # 하드 데이터셋 
    # get_gt_boxes(gt_path) = Ground Truth 박스 정보 로드
    event_num = len(event_list) #event_num = 61
    # event_num: WiderFace 벤치마크에서 평가할 이벤트 개수
    thresh_num = 1000 
    # thresh_num: 1000개의 score 임계값을 사용하여 PR Curve 생성
    settings = ['easy', 'medium', 'hard']
    # settings: WiderFace에서 사용되는 난이도 3가지 (easy, medium, hard)
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    # setting_gts: 난이도 별 Ground Truth 데이터 저장
    aps = []
    for setting_id in range(3):
        # different setting
        gt_list = setting_gts[setting_id]
        # gt_list.shape = (61, 1)
        # gt_list = array([[array([[array([], shape=(0, 1), dtype=uint8)],[array([], shape=(0, 1), dtype=uint8)],[array([], shape=(0, 1), dtype=uint8)],[array([[1],[2],[3],[4],[5]], dtype=uint8)],
        # gt_list: 현재 평가할 난이도의 GT 데이터 가져오기
        # gt_list[0][0][100] = array([array([[2],[3],[4],[6],[9]], dtype=uint8)], dtype=object)
        # gt_list[0][0][100]는 101번째 이미지에서 평가할 5개의 GT 바운딩 박스가 있고, 해당 인덱스들이 [2,3,4,6,9]라는 뜻.
        count_face = 0 # count_face: GT에 포함된 전체 얼굴 개수
        pr_curve = np.zeros((thresh_num, 2)).astype('float') # Precision-Recall 곡선 데이터를 저장하는 배열
        # pr_curve = array([[0., 0.],[0., 0.],[0., 0.],...,[0., 0.],[0., 0.],[0., 0.]])
        # pr_curve.shape = (1000, 2) = (임계값 개수=1000, Precision, recall 2개)
        # 1000개의 confidence threshold마다 precision과 recall값을 저장함 
        # [hard, medium, easy]
        pbar = tqdm.tqdm(range(event_num)) # tqdm.tqdm(): 진행상황을 실시간으로 확인하는 progress bar

        for i in pbar:
            pbar.set_description('Processing {}'.format(settings[setting_id]))
            event_name = str(event_list[i][0][0]) 
            # event_name = '0--Parade'# event_name: 현재 평가 중인 이벤트 이름 (예: street_001, mall_002 등)
            img_list = file_list[i][0] # img_list: 해당 이벤트에 포함된 모든 이미지
            pred_list = pred[event_name] # 현재 이벤트에서 예측된 bounding box 목록
            sub_gt_list = gt_list[i][0] # 해당 이벤트의 GT 얼굴 리스트
            # gt_list에서 모든 Ground Truth를 가져오지 않는 이유는, 이미 난이도별로 해당하는 Ground Truth가 gt중에서도 index로 정해져있기 때문임 
            # img_pr_info_list = np.zeros((len(img_list), thresh_num, 2))
            gt_bbx_list = facebox_list[i][0] #  GT 박스 리스트
            # facebox_list[1][0] = array([[array([[857,  21,  42,  62],[651, 107,  15,  22],

            for j in range(len(img_list)):
                pred_info = pred_list[str(img_list[j][0][0])]
                # pred_info: 현재 이미지에서 예측된 바운딩 박스 정보
                gt_boxes = gt_bbx_list[j][0].astype('float')
                # gt_boxes: 현재 이미지의 Ground Truth 박스 정보
                keep_index = sub_gt_list[j][0]
                # keep_index: 특정 GT 박스를 유지할 인덱스 (일부 GT는 평가 제외 가능)
                count_face += len(keep_index)
                # count_face: 평가에 사용된 GT 얼굴 개수를 누적
                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index-1] = 1
                pred_recall, proposal_list = image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                # pred_recall: 각 예측된 박스에 대해 recall 값 저장
                # proposal_list: 최종적으로 유효한 예측 박스를 나타내는 리스트 (1이면 유효, -1이면 무효)
                _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
                # _img_pr_info = precision, recall 값 
                # pr_info.shape = (1000, 2)
                # 첫 번째 열: Precision 값
                # 두 번째 열: Recall 값
                pr_curve += _img_pr_info
        pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]

        ap = voc_ap(recall, propose)
        aps.append(ap)

    print("==================== Results ====================")
    print("Easy   Val AP: {}".format(aps[0]))
    print("Medium Val AP: {}".format(aps[1]))
    print("Hard   Val AP: {}".format(aps[2]))
    print("=================================================")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="/home/youngoh/retinaface/Pytorch_Retinaface/widerface_evaluate/widerface_txt/") # ./widerface_txt/에서 디버깅안되서 패스로 직접지정변경함
    parser.add_argument('-g', '--gt', default='/home/youngoh/retinaface/Pytorch_Retinaface/widerface_evaluate/ground_truth/') # ./ground_truth/ 에서 디버깅모드 시 상위폴더로 경로가 이동되는 오류로인해 변경함
    

    args = parser.parse_args()
    evaluation(args.pred, args.gt)












