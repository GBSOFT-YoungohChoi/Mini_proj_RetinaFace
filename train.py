from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_mnet, cfg_re50
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import wandb
import random
from dotenv import load_dotenv
# dotenv
load_dotenv() 

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

# WandB 실행 및 설정 추가가

wandb.init(
    entity=os.environ.get("WANDB_ENTITY"),
    project=os.environ.get("WANDB_PROJECT"),
    config={
        "learning_rate": args.lr,  # argparse에서 받은 learning rate 사용
        "architecture": args.network,  # 네트워크 백본 (MobileNetV2 or ResNet50)
        "dataset": args.training_dataset,  # 학습 데이터셋 경로
        "num_workers": args.num_workers,  # DataLoader에서 사용할 워커 수
        "momentum": args.momentum,  # 모멘텀 값
        "resume_net": args.resume_net,  # 체크포인트 파일 (없으면 None)
        "resume_epoch": args.resume_epoch,  # 재학습 시작 epoch
        "weight_decay": args.weight_decay,  # Weight decay 값
        "gamma": args.gamma,  # Learning Rate Scheduler gamma 값
    }
)

# 설정된 WandB 값 출력
print(f"WandB: \n{wandb.config}")

# 이후 Train Loop에서 wandb.log()를 사용하여 실시간 로깅 가능



if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "mobile0.25":
    cfg = cfg_mnet
elif args.network == "resnet50":
    cfg = cfg_re50

rgb_mean = (104, 117, 123) # bgr order ImageNet mean
num_classes = 2 # classification 0 or 1 (face or background)
img_dim = cfg['image_size'] # 640 * 640 fixed size
num_gpu = cfg['ngpu'] # number of GPU
batch_size = cfg['batch_size']  # batch size
max_epoch = cfg['epoch'] # max epoch 
gpu_train = cfg['gpu_train'] #  use gpu to train

num_workers = args.num_workers 
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
print(net)

# Resnet50 backbone can be loaded
if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

# Using dataparallel by multi gpus to train 
if num_gpu > 1 and gpu_train: 
    net = torch.nn.DataParallel(net).cuda() 
else:
    net = net.cuda()

cudnn.benchmark = True

### Change the optimizer
# optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = optim.AdamW(net.parameters(), lr=initial_lr, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    net.train()
    start_time = time.time()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection( training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    for iteration in range(start_iter, max_iter):
        epoch_start_time = time.time() # record epoch start time
        if iteration % epoch_size == 0: 
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 2 == 0 and epoch > cfg['decay2']): # before:['decay1'] -> after: ['decay2']
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))

        epoch_end_time = time.time()  # record epoch end time
        epoch_time = epoch_end_time - epoch_start_time  # calculate epoch time

        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        
        # Wandb 로깅    
        wandb.log({
            "epoch": epoch, 
            "max_epoch": max_epoch,
            "iteration": iteration, 
            "epoch_size": epoch_size, 
            "max_iter": max_iter, 
            "loss_localization": loss_l.item(), 
            "loss_classification": loss_c.item(), 
            "loss_landmark": loss_landm.item(), 
            "learning_rate": lr, 
            "batch_time": batch_time,
            "epoch_time": epoch_time,
            "ETA_seconds": eta, 
        })
    total_train_time = time.time() - start_time
    print(f" Total training time: {str(datetime.timedelta(seconds=int(total_train_time)))}")
    wandb.log({"total_train_time": total_train_time})
    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')
    # torch.save(net.state_dict(), save_folder + 'Final_Retinaface.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
