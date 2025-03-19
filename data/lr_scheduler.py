import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmUpRestarts(_LRScheduler): #스케줄러 기본 클래스 상속
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        """
        optimizer: 학습률을 조정할 옵티마이저 객체
        T_0: 첫 번째 주기의 길이(에포크 수) # T_0는 양의 정수여야 함
        T_mult: 각 주기 이후 주기 길이의 배수 # T_mult는 1 이상의 정수여야 함
        eta_max : learning rate 최댓값 
        T_up : Warm up 시 필요한 epoch수를 지정하며 일반적으로 짧은 epoch 수를 지정합니다  # T_up는 0 이상의 정수여야 함
        gamma : 각 주기마다 최대 학습률(eta_max)에 곱해지는 감소 비율
        last_epoch: 마지막 에포크 인덱스
        """
        if T_0 <= 0 or not isinstance(T_0, int): # T_0가 양수인지 검증 , 0이거나 음수일 경우, 에러발생
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int): # T_mult 가 1이상의 정수인지 검증, 아닌 경우 ValueError
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int): # T_up이 0 이상의 정수인지 검사. 아닌 경우 ValueError
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0 # 첫번째 주기의 길이
        self.T_mult = T_mult # self.T_mult: 주기 길이를 증가시킬 배수
        self.base_eta_max = eta_max # 처음 지정된 최대 학습률을 따로 저장 (주기가 바뀔 때마다 eta_max가 변할 수 있으므로, 원본값 보관)
        self.eta_max = eta_max # eta_max:  매 주기별로 업데이트될 수 있는 “현재” 최대 학습률
        self.T_up = T_up # 웜업 기간
        self.T_i = T_0 # T_i: 현재 주기의 길이 # (초기에는 T_0로 시작)
        self.gamma = gamma # 각 주기마다 eta_max를 감소시키는 비율
        self.cycle = 0 # 현재가 몇 번째 주기인지를 나타내는 카운트 (0부터 시작)
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch) # 부모클래스 호출
        self.T_cur = last_epoch # 현재 주기 내에서 진행된 에포크를 나타냄. -1이면 학습을 시작하지 않았음을 의미.
    
    def get_lr(self):
        if self.T_cur == -1: # 아직 학습이 시작되지 않았으므로 PyTorch 옵티마이저가 가지고 있는 “기본 학습률”(base_lrs)를 그대로 반환합니다.
            return self.base_lrs
        elif self.T_cur < self.T_up: # T_cur가 T_up보다 작은 경우(웜업 단계) - T_cur가 웜업 기간 T_up 내에 있으므로, 선형적으로 학습률을 올립니다
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs] #
        # LR = base_lr + (eta_max - base_lr) * (현재 웜엄 진행된 에포크 = T_cur / 총 웜업기간 = T_up)
        # 현재 웜업이 진행된 에포크가 5이고, 총 웜업기간이 5이고, base_lr = 0.001이고, eta_mx = 0.01 이라고 한다면, 0.001 + (0.01-0.001) * 5/5 이므로 0.001 + 0.009 * 1 이므로 0.01 이 됨 
        # 결과적으로 웜엄period가 끝나면 learning rate == 0.001 에서 목표하는 0.01 되는것임
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    # 웜업을 마친 뒤(self.T_cur >= self.T_up), T_up 이후부터 T_i까지 코사인 곡선으로 학습률을 낮춰갑니다
                    # LR = base lr + self_eta_max(현재 주기의 최대 learning rate) - base_lr  * (1 + cos(pi * 현재에포크 - 웜업기간)/ 현재주기 - 웜업기간)/2
                    # 코사인 함수의 역할 : cos(0) = 1이고 cos(pi) = -1 이므로 코사인 값이 1에서 -1로 변하게 됨 
                    # 이를 활용하여 학습률을 처음에는 높게 설정한 뒤, 점진적으로 낮추는 것이 목적임  -> math.cos(πx)를 사요하게 되면 학습률이 최댓값에서 최솟값으로 급격히 감소하게 되므로 부드럽게 감소를 위해 변형이 필요함
                    # 1을 더하고 2로 나누는 이유 -> (1+1)/2 = 1 == 학습률이 eta_max에서 시작함, (1-1)/2 = 0 학습률이 0으로 수렴함 
                    # 코사인 감소 구간에서 현재 진행된 정도를 나타내는 주기 
                    # 현재 에포크 - 웜엄기간 = T_cur - T_up = 웜업이후 진행된 에포크수
                    # 현재 주기 길이 - 웜업기간 = T_i - T_up  =  웜업이 끝난 후 남은 주기의 길이 
                    # 결과적으로(T_cur - T_up)/(T_i-T_up) = 1이 되어야 cos(π * 1) 이 되어 -> (1 + math.cos(math.pi * 1)) / 2 -> (1 + -1) / 2 -> 0으로 수렴할 수 있게 됨
                    # 다시말하면 (T_cur - T_up)/(T_i-T_up) = 1 -> T_cur = T_i와 같을때 이 식이 성립함 
                    # step() 함수에서 T_cur이 T_i에 도달하면 주기가 종료되고 다음 주기로 넘어가도록 구현되어 있기 때문에 해당 주기가 끝나면 다음 주기로 넘어가고, 다시 Learning rate가 eta_max까지 올라가게됨 
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1 # epoch가 증가함 
            self.T_cur = self.T_cur + 1 # T_cur을 통해 현재 epoch를 표시함 
            if self.T_cur >= self.T_i: # 현재 epoch가 주기를 넘어가게되면 한 사이클이 끝났음을 의미하게됨 
                self.cycle += 1 # 현재 주기의 index에 +1을 해주게됨
                self.T_cur = self.T_cur - self.T_i # 새 사이클이 0부터 시작하도록 T_cur에서 주기 길이를 뺀 나머지로 업데이트
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up 
                # T_mult가 1보다 큰 경우 주기가 점점 길어지며, 웜업 부분 T_up도 고려하여 (self.T_i - self.T_up) * self.T_mult + self.T_up로 계산
        else: # epoch을 사용자가 지정해 준 경우 
            if epoch >= self.T_0:# T_mult가 1이며, epoch가 T_0 이상일 때, 주기가 고정 길이(T_0)이므로, 현재 에포크를 T_0으로 나눈 나머지가 self.T_cur, 몫이 self.cycle.
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:   # T_mult가 1이 아닌 경우, 주기가 매번 T_mult만큼 증가하므로, 현재 epoch가 몇 번째 주기에 속해 있는지 로그(math.log)를 통해 계산.
                        # self.cycle과 self.T_cur를 각각 업데이트하며, self.T_i도 새 주기의 길이로 설정
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult)) 
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else: # 아직 첫 번째 주기도 끝나지 않았으므로, T_i는 여전히 T_0이고, T_cur = epoch로 설정.
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        # 주기 증가 시 최대 학습률 감소 -> eta_max를 업데이트함 -> eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        # last_epoch 업데이트: 주어진 epoch(또는 자동으로 계산된 값)을 반영. 소수점 입력이 있을 수 있으므로 math.floor 사용
        # 배치단위의 학습으로 인해 e.g., 5번째 에포크의 50번째 배치일 경우, epoch = 5.5와 같이 소수점을  포함하는 값으로 표현됨, 따라서 math.floor(epoch)을 사용함 
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr