from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox # PriorBox 생성 함수
from utils.nms.py_cpu_nms import py_cpu_nms # NMS(Non-Maximum Suppression) 적용 함수
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm # 바운딩 박스 및 랜드마크 디코딩 함수
from utils.timer import Timer
from dotenv import load_dotenv
import time
load_dotenv() 

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='/home/youngoh/retinaface/Pytorch_Retinaface/weights/Adam_epoch500/mobilenet0.25_epoch_300.pth', # ./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50') # default='resnet50'
parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=False, help='show detection results')
parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
args = parser.parse_args()

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' 
    # module. 접두사를 제거하여 모델이 정상적으로 로드될 수 있도록 정리하는 역할을 함'''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # 사전 학습된 모델 가중치를 불러와서 현재 모델에 적용하는 함수 
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device() # GPU에 가중치 로드
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device)) # 모델 가중치 로드
    if "state_dict" in pretrained_dict.keys(): 
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    # pretrained_dict = {'body.stage1.0.0.weight': tensor([[[[ 0.0015,  0.0219, -0.0519],[-0.3598,  0.0413,  0.1013],...07,  0.4092,  0.0314]]]], device='cuda:0'), 
    # layer name과 weight가 저장됨 
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False) # model = RetinaFace model
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False) # 모델 평가 phase

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    model_load_start = time.time()
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu) # 모델 로드
    net.eval()
    model_load_end = time.time()
    print('Finished loading model!')
    print(net)

    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    # testing dataset

    testset_folder = args.dataset_folder # 테스트 데이터셋 폴더 경로 설정 #'./data/widerface/val/images/'
    testset_list = args.dataset_folder[:-7] + "wider_val.txt" # './data/widerface/val/wider_val.txt'

    with open(testset_list, 'r') as fr:
        test_dataset = fr.read().split() # 파일을 읽고 개별 이미지 경로를 리스트로 저장
    num_images = len(test_dataset) # 테스트 데이터 개수 3226 # 테스트할 총 이미지 개수 저장


    _t = {'forward_pass': Timer(), 'misc': Timer()} # 추론 시간 측정을 위한 타이머 객체 생성
    total_prediction_start = time.time()
    # testing begin
    for i, img_name in enumerate(test_dataset): # 테스트 데이터셋을 하나씩 불러옴
        image_path = testset_folder + img_name # 개별 이미지 파일 경로 생성
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR) # OpenCV를 사용하여 컬러 이미지 읽기
        img = np.float32(img_raw) # 이미지를 `float32` 타입으로 변환 (딥러닝 모델 입력을 위해 필요)

        # testing scale
        target_size = 1600 # 최소 축 크기의 목표 크기 설정
        max_size = 2150 # 대 축 크기 설정 (너무 큰 이미지를 방지)
        im_shape = img.shape # 이미지의 현재 크기 가져오기 (height, width, channel)
        im_size_min = np.min(im_shape[0:2]) # height, width 중 작은 값 선택 (작은 축)
        im_size_max = np.max(im_shape[0:2]) # height, width 중 큰 값 선택 (큰 축)
        resize = float(target_size) / float(im_size_min) # 작은 축을 `target_size`에 맞추도록 비율 계산
        # prevent bigger axis from being more than max_size:
        # 큰 축이 `max_size`를 초과하지 않도록 조정
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        # 원본 크기 유지 옵션이 활성화되었을 경우 `resize=1`로 설정
        if args.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
            # `resize` 배율을 사용하여 이미지를 크기 조정 (가로 `fx`, 세로 `fy` 비율 적용)
            # `cv2.INTER_LINEAR` 보간법 사용 (보통 이미지 크기 조절 시 적절한 방법)
        
        im_height, im_width, _ = img.shape # 이미지 크기(height, width)와 채널 개수 추출
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # 바운딩 박스를 정규화된 좌표에서 원본 이미지 좌표로 변환할 때 사용할 스케일 값
        # scale = [width, height, width, height] 값으로 변환
    
        img -= (104, 117, 123) # 이미지 정규화: RGB 평균값(104, 117, 123)을 빼서 데이터 중심을 0 근처로 이동
        img = img.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).unsqueeze(0) # numpy 배열을 PyTorch 텐서로 변환하고 batch 차원을 추가
        img = img.to(device)
        # img.shape = torch.Size([1, 3, 562, 1024]) #이미지 크기 그대로 들어옴옴
        scale = scale.to(device)
        # scale.shape = torch.Size([4]) 
        # scale = tensor([1024.,  562., 1024.,  562.], device='cuda:0') # 원본사이즈를 보관함 
        _t['forward_pass'].tic() # 모델 추론 시간 측정을 시작
        loc, conf, landms = net(img)  # forward pass # RetinaFace 모델을 통해 얼굴 검출 수행 (추론)
        _t['forward_pass'].toc() # 추론 시간 측정 종료
        _t['misc'].tic() # 기타 후처리 작업 시간 측정 시작
        priorbox = PriorBox(cfg, image_size=(im_height, im_width)) # PriorBox 생성 (사전에 정의된 anchor box를 생성하는 클래스)
        priors = priorbox.forward()
        # priors.shape = torch.Size([23936, 4]) # 앵커 생성 -> input_dim에 따라 다르므로 학습때보다 많아질 수 있음 
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        # boxes.shape = (129, 4)
        
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        # 신뢰도(conf)에서 배경(0)과 객체(1) 중 얼굴(1)의 신뢰도 점수를 가져옴

        # scores.shape = (23936,)
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']) # 랜드마크 위치 정보도 디코딩하여 실제 좌표로 변환
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2], # 랜드마크 위치를 원래 이미지 크기에 맞게 조정
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        # 신뢰도 점수가 설정한 임계값(args.confidence_threshold)보다 높은 박스만 선택
        inds = np.where(scores > args.confidence_threshold)[0] # 임계값을 초과하는 인덱스만 가져옴
        boxes = boxes[inds] # 해당 인덱스의 박스만 선택
        landms = landms[inds] # 해당 인덱스의 랜드마크만 선택
        scores = scores[inds] # 해당 인덱스의 신뢰도 점수만 선택




        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args.top_k]
        # 신뢰도(confidence score)를 기준으로 내림차순 정렬 (가장 높은 점수를 가진 박스가 먼저 오도록 배열)
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # 정렬된 순서에 맞춰 boxes, landms, scores도 동일한 순서로 정렬
        # args.top_k 값이 있다면, 상위 top_k 개수만 유지하도록 제한할 수도 있음

        # do NMS 중복박스 제거 non-maximum suppression
        # 여러 개의 바운딩 박스 중 신뢰도(confidence score)가 높은 것만 남기고 나머지는 제거하는 과정
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # 신뢰도 점수가 높은 박스를 남기고 겹치는 영역이 많은 박스를 제거
        keep = py_cpu_nms(dets, args.nms_threshold) # NMS에서 겹침 허용 기준 (일반적으로 0.3~0.5 사용)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # 유지할 박스만 keep 변수에 저장한 후, 해당 박스들만 남김

        # keep top-K faster NMS
        # dets = dets[:args.keep_top_k, :]
        # landms = landms[:args.keep_top_k, :]
        # 탐지된 박스 개수를 keep_top_k 개수로 제한할 수 있도록 주석 처리됨
        # 필요하다면 args.keep_top_k를 설정하여 너무 많은 박스가 유지되는 것을 방지 가능


        dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()
        # 바운딩 박스 정보(dets)와 랜드마크(landms)를 하나로 합침
        # _t['misc'].toc(): Miscellaneous 처리 시간 측정 종료
        # --------------------------------------------------------------------
        save_name = args.save_folder + img_name[:-4] + ".txt"
        dirname = os.path.dirname(save_name)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        # 결과 파일 저장 경로 설정 (args.save_folder 하위 폴더에 저장)
        # os.makedirs()를 사용해 폴더가 없으면 자동으로 생성

        with open(save_name, "w") as fd:
            bboxs = dets
            file_name = os.path.basename(save_name)[:-4] + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            fd.write(file_name)
            fd.write(bboxs_num)
            # 탐지된 바운딩 박스 정보를 .txt 파일로 저장
            # 첫 줄에 이미지 파일 이름, 두 번째 줄에 탐지된 박스 개수 저장
            for box in bboxs:
                x = int(box[0])
                y = int(box[1])
                w = int(box[2]) - int(box[0])
                h = int(box[3]) - int(box[1])
                confidence = str(box[4])
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) + " " + confidence + " \n"
                fd.write(line)
            # 탐지된 바운딩 박스 좌표를 저장
            # (x, y): 좌상단 좌표
            # (w, h): 바운딩 박스의 너비 및 높이
            # confidence: 신뢰도 점수
        print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(i + 1, num_images, _t['forward_pass'].average_time, _t['misc'].average_time))
        #탐지 진행 상황을 터미널에 출력 
        # i + 1: 현재 탐지 중인 이미지 번호
        # num_images: 총 이미지 개수
        # forward_pass_time: 네트워크 예측(Forward Pass) 소요 시간
        # misc: 그 외 처리 소요 시간 (NMS, 결과 저장 등)

        # save image
        if args.save_image:
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                # 탐지된 바운딩 박스의 confidence score가 vis_thres(시각화 임계값)보다 낮으면 무시
                # 즉, 너무 낮은 신뢰도 점수는 시각화하지 않음
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
                #탐지된 얼굴을 빨간색 박스로 그림
                # (b[0], b[1]): 좌상단 좌표
                # (b[2], b[3]): 우하단 좌표
                # 신뢰도 점수를 흰색 텍스트로 표시

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
                # 탐지된 얼굴 랜드마크(눈, 코, 입)를 원(circle)으로 시각화
                # 각각 다른 색상으로 표현
            # save image
            if not os.path.exists("./results/default/"):
                os.makedirs("./results/default/")
            name = "./results/default/" + str(i) + ".jpg"
            cv2.imwrite(name, img_raw)

    total_prediction_end = time.time()
    print(f"Model loading time: {model_load_end - model_load_start:.4f} seconds")
    print(f"Total inference time for {num_images} images: {total_prediction_end - total_prediction_start:.4f} seconds")
    print(f"Average inference time per image: {(total_prediction_end - total_prediction_start) / num_images:.4f} seconds")