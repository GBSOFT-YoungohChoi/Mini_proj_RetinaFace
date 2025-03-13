import cv2
import numpy as np
import random
from utils.box_utils import matrix_iof


def _crop(image, boxes, labels, landm, img_dim):
    height, width, _ = image.shape
    # image shape -> height, width 할당 
    # e.g., 771 * 1024 * 3 image shape -> height = 771, width = 1024
    pad_image_flag = True

    for _ in range(250):
        """
        if random.uniform(0, 1) <= 0.2:
            scale = 1.0
        else:
            scale = random.uniform(0.3, 1.0)
        """
        PRE_SCALES = [0.3, 0.45, 0.6, 0.8, 1.0] # 다양한 크기로 원본이미지를 잘라줌 
        scale = random.choice(PRE_SCALES)
        # 랜덤하게 scale을 고름름
        short_side = min(width, height)
        # short_side = 771, 1024 중에 작은 771값 할당
        # width, height값중에 최저값을 고름 
        w = int(scale * short_side)
        # 랜덤하게 고른 scale에 따라서 더 작은 값을 곱해줌 
        h = w # 정사가형 형태의 크롭 영역을 유지함 

        if width == w: # 크롭할 너비와 원본 너비가 동일할 경우, 왼쪽끝(l=0)에서 시작하게됨
            l = 0
        else: # 크롭할 너비와 원본 너비가 동일하지 않을 경우, 원본이미지 width에서 w를 빼고, 랜덤한 시작점을 선택함 
            l = random.randrange(width - w)
        if height == h: # 크롭할 너비와 원본 너비가 동일할 경우, 왼쪽끝(l=0)에서 시작하게됨
            t = 0 
        else: # 크롭할 너비와 원본 너비가 동일하지 않을 경우, 원본이미지 height에서 h를 빼고, 랜덤한 시작점을 선택함 
            t = random.randrange(height - h) 
        roi = np.array((l, t, l + w, t + h))
        # l은 width crop 시작지점, t는 height 시작지점, l + w값은 width 끝지점, t + h값 은 height 끝지점
        # roi = array([196,   0, 967, 771]) --> crop 영역
        value = matrix_iof(boxes, roi[np.newaxis])
        # boxes = array([[158.,  70., 332., 300.],[468.,  90., 710., 408.]])
        # boxes.shape = (2, 4)
        # roi[np.newaxis] = array([[196,   0, 967, 771]])
        # roi[np.newaxis].shape = (1, 4)
        # matrix_iof 의 출력값 --> area_i / np.maximum(area_a[:, np.newaxis], 1)
        # value = array([[0.7816092], [1.       ]])
        flag = (value >= 1)# IOF 값이 1 이상인지 확인 (1이라는 의미는 GT 박스가 ROI 영역에 완전히 포함되어 있다는 의미)
        # flag = array([[False], [ True]])
        if not flag.any(): # IOF 값이 1 이상인 GT 박스가 하나도 없으면, 즉, 크롭된 영역 안에 GT 박스(얼굴)이 없으면 현재 크롭을 버리고 다음 크롭을 시도
            continue # 크롭된 이미지 안에 GT가 포함되지 않으면 학습데이터로 적합하지 않기 때문에 버려야함, 데이터 품질을 위해 버림림

        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        # boxes[:, :2] → 박스의 좌상단 좌표 (xmin, ymin)
        # centers =  (xmin, ymin)과 (xmax, ymax)를 더한 후 2로 나누면, 박스의 중심점 좌표(cx, cy)가 됨
        mask_a = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
        # roi[:2] = 크롭이미지 좌상단 좌표 = [xmin_roi, ymin_roi] (ROI의 좌상단 좌표)
        # roi[2:] = 크롭이미지 우하단 좌표 = [xmax_roi, ymax_roi] (ROI의 우하단 좌표)
        # roi[:2] < centers = array([[ True,  True]]) bounding box의 중심이 ROI의 왼쪽 위 좌표보다 더 오른쪽 & 아래에 있는지 판별 -> 안에 있으면 True True
        # centers < roi[2:] = array([[ True,  True]]) bounding box의 중심이 ROI의 오른쪽 아래 좌표보다 더 왼쪽 & 위에 있는지 판별 -> 안에 있으면 True True
        # np.logical_and([True, True], [True, True]) 이면 True를 출력함 
        # mask_a = array([ True])
        boxes_t = boxes[mask_a].copy()
        # boxes = array([[490., 348., 526., 412.]])
        # boxes[mask_a] = array([[490., 348., 526., 412.]])
        # boxes[False] = array([], shape=(0, 1, 4), dtype=float64) 
        labels_t = labels[mask_a].copy()
        # labels = array([1.])
        # labels[mask_a] = array([1.])
        # labels_t = array([1.])

        landms_t = landm[mask_a].copy()
        # landm = array([[514.406, 375.438, 518.062, 375.031, 522.531, 387.625, 515.625, 398.188, 519.688, 397.781]])
        # landm = [얼굴개수, landmark 10개]로 구성됨
        # landm.shape = (6, 10) 이었는데, mask_a = array([ True,  True, False, False,  True,  True]) 여서 2개가 False이므로 지워짐 따라서 
        # landms_t.shape = (4, 10) # 얼굴이 4개 존재함 
        landms_t = landms_t.reshape([-1, 5, 2])
        # landm = array([[514.406, 375.438], [518.062, 375.031], [522.531, 387.625], [515.625, 398.188], [519.688, 397.781]]])
        
        if boxes_t.shape[0] == 0:
            continue

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]
        # roi = [좌상단x, 좌상단y, 우하단x, 우하단y]
        # roi[1]:roi[3] → 높이, roi[0]:roi[2] → 너비 범위로 원본 이미지를 잘라냄
        # image_t = crop된 이미지 부분
        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        # 원래 bounding box의 좌상단 좌표 (x_min, y_min)을 roi[:2] = (l, t)과 비교
        # 만약 bounding box가 crop된 영역 바깥에 있으면 roi[:2] 값으로 조정
        boxes_t[:, :2] -= roi[:2]
        # 좌상단 좌표를 crop된 이미지 기준으로 이동
        # roi[:2] 값을 빼서 좌표계를 (0, 0) 기준으로 정렬
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
        # boxes_t[:, 2:] = bounding box의 우하단 좌표 (x_max, y_max)
        # roi[2:] = (l + w, t + h)을 기준으로 bounding box가 crop 영역을 넘지 않도록 조정
        boxes_t[:, 2:] -= roi[:2]
        # 우하단 좌표도 crop된 이미지 기준으로 변환
        # landm
        landms_t[:, :, :2] = landms_t[:, :, :2] - roi[:2]
        # 랜드마크 좌표 landms_t도 bounding box처럼 crop된 이미지 기준으로 변환
        landms_t[:, :, :2] = np.maximum(landms_t[:, :, :2], np.array([0, 0]))
        # 랜드마크 좌표가 이미지 바깥으로 나가는 경우, 0으로 고정 
        # e.g., landms_t = [[[−5, 20], [30, 40], [10, 50], [40, 60], [50, 70]]] → [[[0, 20], [30, 40], [10, 50], [40, 60], [50, 70]]]
        landms_t[:, :, :2] = np.minimum(landms_t[:, :, :2], roi[2:] - roi[:2])
        # crop된 이미지 크기보다 클 경우 최대값을 제한
        landms_t = landms_t.reshape([-1, 10])
        # reshape([-1, 10])을 통해 [num_landmarks, 10]으로 변환 (2D → 1D)
        # landms_t = [[[20, 30], [30, 40], [10, 50], [40, 60], [80, 90]]] → [[20, 30, 30, 40, 10, 50, 40, 60, 80, 90]]

	# make sure that the cropped image contains at least one face > 16 pixel at training image scale
        b_w_t = (boxes_t[:, 2] - boxes_t[:, 0] + 1) / w * img_dim
        # boxes_t[:, 2] = array([512., 576.,  66., 381.])
        # boxes_t[:, 0] + 1 = array([498., 561.,  52., 345.])
        # w = width
        # img_dim = 640
        # x_max - x_min만 사용한 경우 -> width = 10 - 5 = 5  (실제 너비보다 1픽셀셀 작음)
        # b_w_t = array([14.99267936, 15.92972182, 14.99267936, 35.60761347])
        # b_w_t.shape = (4,)
        b_h_t = (boxes_t[:, 3] - boxes_t[:, 1] + 1) / h * img_dim
        # boxes_t = array([[  0., 319.,  93., 460.], [129., 288., 143., 302.]]) 일때
        # boxes_t[:, 3] = array([460., 302.])
        # boxes_t[:, 1] = array([319., 288.])
        # b_h_t = 바운딩 박스의 높이 정규화 및 훈련 이미지 크기에 맞게 변환하는 과정 
        # boxes_t[:, 3] - boxes_t[:, 1] 값이 y_max - y_min이고 Bbox의 높이가 됨
        # y_max - y_min 를 수행하게 되면 출력값이 실제값보다 1픽셀 작은 값이 도출되어 "+1"을 해줌 
        # h 인 크롭된 이미지 높이로 정규화해주고, 이를 학습 이미지 크기에 맞춰서 변환해줌 -> 결국 학습이미지로 변환하는 과정임
        mask_b = np.minimum(b_w_t, b_h_t) > 0.0
        # 결국 b_w_t, b_h_t는 학습이미지 크기 640에 맞춘 bbox의 너비와 높이임, 크롭된 이미지를 기준으로 정규화된 값임  
        # b_w_t와 b_h_t중에서 더 작은값(0보다 큰값)을 선택함
        # 너비와 높이가 0이하인 경우 필터링 진행 
        # 크롭과정에서 일부 바운딩 박스가 너무 작아질 수 있음 
        # 결국 이과정은 0 이하인 바운딩박스를 가려내기위한 코드임임
        boxes_t = boxes_t[mask_b]
        labels_t = labels_t[mask_b]
        landms_t = landms_t[mask_b]
        # 0이하인 박스, 레이블, landmark를 걸ㄹ냄 
        if boxes_t.shape[0] == 0:
            continue

        pad_image_flag = False

        return image_t, boxes_t, labels_t, landms_t, pad_image_flag
    return image, boxes, labels, landm, pad_image_flag


def _distort(image): # 밝기, 대비, 채도, 색조를 랜덤하게 왜곡하는 역할

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta # tmp = image.shape과 같음 
        # pixel값에 alpha 배 + beta 추가값 적용

        tmp[tmp < 0] = 0 # 이미지 픽셀값이 0보다 작아지는 상황을 제한함
        tmp[tmp > 255] = 255 # 이미지 픽셀값이 255보다 커지는 상황을 제한함
        image[:] = tmp 

    image = image.copy() # 원본이미지 놔두고고

    if random.randrange(2):

        #brightness distortion
        if random.randrange(2): # 50퍼센트의 확률로 실행 
            _convert(image, beta=random.uniform(-32, 32)) # 밝기를 -32 ~ +32 범위에서 랜덤하게 조정정

        # contrast distortion (밝은 부분과 어두운 부분의 명암차이)
        # 대비가 높을수록 밝은부분은 더 밝게, 어두운 부분은 더 어두워져서 차이가 뚜렷해짐짐
        # 대비가 낮을 수록 전체적으로 균일한 밝기를 가지며 흐릿해짐 
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5)) # 대비를 0.5배 ~ 1.5배로 조정
        
        #saturation distortion
        # HSV 색공간의 채도 공간간
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        # image[:, :, 0] =  H (Hue, 색조)
        # hsv_image[:, :, 1] = S (Saturation, 채도)
        # hsv_image[:, :, 2] = V (Value, 명도)
        
        # 이미지 색상의 채도를 조절하는 과정
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:

        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _expand(image, boxes, fill, p):
    if random.randrange(2): # 50% 확률로 이미지를 확장하지 않고 원본 그대로 반환
        return image, boxes

    height, width, depth = image.shape

    scale = random.uniform(1, p) # scale: 1 ~ p 사이의 랜덤 값, # 이미지를 p배 크기까지 확장할 수 있음
    w = int(scale * width)
    h = int(scale * height)

    left = random.randint(0, w - width) # 확장된 이미지(w, h) 내에서 원본 이미지의 시작 위치를 랜덤으로 설정
    top = random.randint(0, h - height) # 확장된 이미지(w, h) 내에서 원본 이미지의 시작 위치를 랜덤으로 설정

    boxes_t = boxes.copy() # 기존 bounding box 좌표를 확장된 이미지에 맞게 이동
    boxes_t[:, :2] += (left, top)# 기존 bounding box 좌표를 확장된 이미지에 맞게 이동
    boxes_t[:, 2:] += (left, top)# 기존 bounding box 좌표를 확장된 이미지에 맞게 이동
    expand_image = np.empty(
        (h, w, depth),
        dtype=image.dtype)
    expand_image[:, :] = fill
    # (h, w, 3) 크기의 빈 배열을 생성
    # fill: 배경 색상 (일반적으로 RGB 평균값)
    expand_image[top:top + height, left:left + width] = image
    # 확장된 이미지에 원본 이미지를 (top, left) 위치에 복사
    image = expand_image

    return image, boxes_t
    # 확장된 이미지 및 새로운 bounding box 좌표 반환

def _mirror(image, boxes, landms):
    _, width, _ = image.shape 
    #image.shape = (1366, 1024, 3)
    # width = 1024
    if random.randrange(2):# 0 또는 1을 랜덤으로 반환 -> 50 % 확률로 이미지 반전
        image = image[:, ::-1]
        #image[:, ::-1].shape = (1366, 1024, 3)
        # 이미지르 좌우로 반전시킴 
        boxes = boxes.copy() # boxes를 보존하여 원본데이터가 변경되지 않도록 함
        # boxes.shape = (2, 4)
        boxes[:, 0::2] = width - boxes[:, 2::-2]
        # boxes[:, 0::2] → x 좌표들 (x_min, x_max)을 가져옴
        # boxes[:, 2::-2] → x_max, x_min을 가져와서 반전된 위치를 계산

        # landm
        landms = landms.copy()
        landms = landms.reshape([-1, 5, 2])
        # 랜드마크 좌표를 조정하기 위해 배열을 3D 형태로 변경
        landms[:, :, 0] = width - landms[:, :, 0]
        # 랜드마크의 x 좌표를 반전된 이미지에 맞게 조정
        tmp = landms[:, 1, :].copy() # 첫 번째(left eye)와 두 번째(right eye) 랜드마크의 위치를 교환.
        landms[:, 1, :] = landms[:, 0, :] # 눈위치 교체
        landms[:, 0, :] = tmp # 눈위치 교체체
        tmp1 = landms[:, 4, :].copy() # 네 번째(right mouth corner)와 다섯 번째(left mouth corner) 랜드마크의 위치를 교환
        landms[:, 4, :] = landms[:, 3, :]# 네 번째(right mouth corner)와 다섯 번째(left mouth corner) 랜드마크의 위치를 교환
        landms[:, 3, :] = tmp1# 네 번째(right mouth corner)와 다섯 번째(left mouth corner) 랜드마크의 위치를 교환
        landms = landms.reshape([-1, 10]) #[-1, 5, 2] → [-1, 10] 형태로 변경 (1D 배열로 펼침)

    return image, boxes, landms


def _pad_to_square(image, rgb_mean, pad_image_flag):
    if not pad_image_flag: # pad_image_flag가 없으면 image.shape = (640, 640, 3)으로 출력됨 
        return image
    height, width, _ = image.shape
    # image.shape = (1250, 1024, 3)
    long_side = max(width, height) # 긴쪽을 찾음 
    image_t = np.empty((long_side, long_side, 3), dtype=image.dtype)
    # np.empty(shape, dtype) -> np.zeros()와 다르게, 초기화 되지 않는 배열을 생성하는 Numpy 함수
    # 배열의 값을 0으로 초기화 하지 않고 메모리 ㅏㅇ의 기존 데이터를 유지함
    # (long_side, long_side, 3) -> long_side 크기의 정사각형 이미지, 3채널 (RGB)
    # 빈 image_t를 생성함 
    # e.g., 크기가 큰쪽으로 사이즈를 맞춰서 배열을 만듦
    image_t[:, :] = rgb_mean
    # 전체 배경을 rgb_mean값으로 채움
    image_t[0:0 + height, 0:0 + width] = image
    # image_t의 크기를 원본이미지의 크기와 높이로 조절함 -> 원본이미지를 정사각형 이미지로 변환하는 과정이며, 정사각형이 맞지 않는 부분은 rgb_mean값으로 패딩하게됨  
    return image_t


def _resize_subtract_mean(image, insize, rgb_mean): 
    # 입력 이미지를 특정 크기로 리사이징하고 평균 RGB 값을 빼준 후, 차원 순서를 변경하는 전처리 과정
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    # OpenCV에서 제공하는 5가지 보간법 중에서 하나를 랜덤으로 선택 
    # cv2.INTER_LINEAR (양선형 보간, 기본값)
    # cv2.INTER_CUBIC (3차 보간, 부드러운 결과)
    # cv2.INTER_AREA (픽셀 영역 관계 기반, 축소할 때 유리)
    # cv2.INTER_NEAREST (최근접 이웃 보간, 빠르지만 품질이 낮음)
    # cv2.INTER_LANCZOS4 (Lanczos 보간, 고품질)
    # 다양성을 증가시킴
    interp_method = interp_methods[random.randrange(5)]

    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    # 이미지 사이즈를 640 * 640으로 변형 
    image = image.astype(np.float32)
    # 이미지를 32비트 실수형으로 변환 
    image -= rgb_mean
    # 평균을 빼주게되면 데이터의 중심이 0에 가까워지므로 학습이 안정화됨 -> 정규화과정 
    # 나눠줄때는 0~1의 값에 맞추기 위해서 나눠주고, 빼줄때는 평균값을 0으로 맞춰주기 위해서 빼줌 
    return image.transpose(2, 0, 1) # openCV입력형태(H, W, C) -> torch입력형태(C, H, W)


class preproc(object): # (object): python2와 호환성을 고려한 표현, python3는 생략해도 자동으로 object상속함

    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert targets.shape[0] > 0, "this image does not have gt" 
        # targets.shape[0] > 0이 False이면 "this image does not have gt"라는 오류 메시지를 띄우고 코드 실행을 멈춥니다 -> gt값이 없는것은 들어가지 않게 바로 실행을 중단함 
        # targets.shape = (6, 15) (2개의 GI값을 가지고 있음 ) --> boxes.shape = (6, 4) --> # labels.shape = (6,) --> # landm.shape = (6, 10)
        boxes = targets[:, :4].copy()
        # boxes.shape = (6, 4)
        labels = targets[:, -1].copy()
        # labels.shape = (6,)
        landm = targets[:, 4:-1].copy()

        # 이미지와 레이블이 같이 들어오게됨
        image_t, boxes_t, labels_t, landm_t, pad_image_flag = _crop(image, boxes, labels, landm, self.img_dim)
        # crop을 진행함, 랜덤하게 각각 1/5 확률로 [0.3, 0.45, 0.6, 0.8, 1.0]의 배로 이미지를 잘라줌
        # 크롭된 이미지안에 적절한 바운딩박스가 포함될 때까지 250번을 수행함 
        # 얼굴이 중심이 크롭안에 있는지 없는지를 확인함 
        image_t = _distort(image_t) # 크롭된 이미지에 대해서 색상 왜곡 진행(밝기, 대비, 채도, 색조 변형)
        image_t = _pad_to_square(image_t,self.rgb_means, pad_image_flag)
        # 색상왜곡된 이미지를 정사각형 형태의 이미지로 패딩을 넣어줌, 이때 배경을 rgb_mean값으로 변경함 
        image_t, boxes_t, landm_t = _mirror(image_t, boxes_t, landm_t) # 50퍼센트의 확률로 좌우반전시킴

        height, width, _ = image_t.shape
        image_t = _resize_subtract_mean(image_t, self.img_dim, self.rgb_means)
        # _resize_subtract_mean를 통해 이미지픽셀 - rgb_mean이후, 640x640으로 맞춰줌
        boxes_t[:, 0::2] /= width
        # boxes_t = array([[0.00630915, 0.03785489, 0.08832808, 0.13880126]])
        # boxes_t[:, 0::2] = array([[0.00630915, 0.08832808]])
        # x_min,x_max값을 너비로 나눠주어 비율로 나타내줌 
        boxes_t[:, 1::2] /= height
        # boxes_t[:, 1::2] = array([[0.03785489, 0.13880126]])
        # y_min, y_max값을 높이로 나눠주어 비율로 나타내줌 
        landm_t[:, 0::2] /= width
        # landm_t = array([[ 8.429, 21.866, 20.134, 23.098, 10.482, 28.438,  8.223, 37.473, 15.821, 37.473]])
        # 랜드마크의 x좌표들도 모두 너비로 나눠주어, 640 * 640의 위치를 잡아줌
        landm_t[:, 1::2] /= height
        # 랜드마크의 y좌표들도 모두 너비로 나눠주어, 640 * 640의 위치를 잡아줌


        labels_t = np.expand_dims(labels_t, 1)
        # hstak을 사용하여 결합을 해주어야하는데 labels_t의 shape이 맞지 않아 결합이 되지 않음
        # hstack이란  바운딩 박스 좌표, 랜드마크 좌표, 라벨을 하나의 배열로 합치는 코드
        targets_t = np.hstack((boxes_t, landm_t, labels_t))
        # np.hstack()은 수평(열 방향)으로 배열을 결합
        # boxes_t → 바운딩 박스 좌표 (N, 4)
        # landm_t → 랜드마크 좌표 (N, 10)
        # labels_t → 객체 라벨 (N, 1)
        return image_t, targets_t 
        # image_t.shape = (3, 640, 640) 출력
        # targets_t.shape = (1, 15) 출력
