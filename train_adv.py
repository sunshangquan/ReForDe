from __future__ import print_function
import torch.optim as optim
import argparse
import os, sys, cv2
from os.path import join as opjoin
import torch
from importlib import import_module
import random
import re
import time
import statistics
import torch.nn.functional as F
# from reforde import reforde
import utils.pytorch_ssim as pytorch_ssim
from natsort import natsorted
from PIL import Image
import numpy as np
from tensorboardX import SummaryWriter
from loss import L1_Charbonnier_loss

sys.path.append('/home1/ssq/proj3/reforde2/models/detect_models/')
sys.path.append('/home1/ssq/proj3/reforde2/utils/')
from utils.dataset_utils.preprocessing import letterbox_image_padded, letterbox_image_padded_PIL
from utils.misc_utils.visualization import visualize_detections, visualize_detections_cv2
from utils.attack_utils.attack import tog_mislabeling

parser = argparse.ArgumentParser(description="PyTorch Test")
parser.add_argument('--lq_path', type=str, default='../VOC_fog/train/hazy', help='Path of the validation dataset')
parser.add_argument('--gt_path', type=str, default='../VOC_fog/train/clean', help='Path of the validation dataset')
parser.add_argument('--an_path', type=str, default='../VOC_fog/train/an', help='Path of the validation dataset')
parser.add_argument('--model_res', type=str, default='GridDehaze', help='filename of the training models')
parser.add_argument('--task', type=str, default='dehaze', help='filename of the training models')

parser.add_argument('--model_det', type=str, default='yolov3', help='filename of the training models')

parser.add_argument('--det_size', type=float, default=640, help='')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training settings
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="pretrained_weights/restore_models/MSBDN/model.pkl", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--train_step", type=int, default=1, help="Activated gate module")
# parser.add_argument("--clip", type=float, default=0.25, help="Clipping Gradients. Default=0.1")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument('--tmppath', type=str, default='tmp2', help='')

training_settings = {'nEpochs': 10, 'lr_decay': 0.1}



####### Global Variables #######
opt = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids[0]

assert opt.model_res.lower() in ['msbdn', 'griddehaze', 'sm']
assert opt.model_det.lower() in ['yolov3', 'frcnn']
os.makedirs('./%s' % opt.tmppath, exist_ok=True)
os.makedirs('./%s/a' % opt.tmppath, exist_ok=True)
print(opt.resume)
if opt.gt_path != '':
#    criterion = torch.nn.L1Loss(size_average=True)
    criterion = L1_Charbonnier_loss()
    criterion = criterion.to(device)
else:
    criterion = None

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')

str_ids = opt.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)

##### log path #####
opt.log_path = opjoin('./log', opt.model_res+"_"+opt.model_det+"_adv_dlr5_lr5")
os.makedirs(opt.log_path, exist_ok=True)
os.makedirs(opjoin(opt.log_path, 'weight'), exist_ok=True)
##### Load Model #####
print("===> Loading restoration model {}".format(opt.model_res))
if opt.model_res.lower() == 'msbdn':
    MSBDN = import_module("models.restore_models.MSBDN."+opt.model_res.upper())

    model = MSBDN.Net()
    model.load_state_dict(torch.load(opt.resume)['state_dict'], strict=True)
elif opt.model_res.lower() == 'griddehaze':
    GridDehaze = import_module("models.restore_models.GridDehaze.model")
    model = GridDehaze.GridDehazeNet()
    if 'griddehaze' not in opt.resume.lower():
        opt.resume = './pretrained_weights/restore_models/GridDehaze/outdoor_haze_best_3_6'
    checkpoint_ = torch.load(opt.resume)
    try:
        model.load_state_dict(checkpoint_['state_dict'], strict=True)
    except:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint_.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
elif opt.model_res.lower() == 'sm':
    from small_model import net
    model = net()
    model.load_state_dict(torch.load('./pretrained_weights/restore_models/sm/sm.pkl')['state_dict'])
print("===> Loading detection model {}".format(opt.model_det))
if opt.model_det.lower() == 'yolov3':
    import tensorflow as tf
    config = tf.ConfigProto(allow_soft_placement=True)
#    config.gpu_options.visible_device_list = opt.gpu_ids[0]
    config.gpu_options.per_process_gpu_memory_fraction = 1. # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.Session(config=config)
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    print("GPUS", gpus)
#    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    yolov3 = import_module("models.detect_models." + opt.model_det.lower())
    detector = yolov3.YOLOv3_Darknet53(weights='pretrained_weights/detect_models/yolov3_best.h5', model_img_size=(opt.det_size, opt.det_size))
elif opt.model_det.lower() == 'frcnn':
    frcnn = import_module("models.detect_models." + opt.model_det.lower())
    detector = frcnn.FRCNN(model_img_size=(opt.det_size, opt.det_size)).to(device).load('pretrained_weights/detect_models/FRCNN_worst.pth' )
    from models.detect_models.frcnn_trainer import FasterRCNNTrainer
    trainer = FasterRCNNTrainer(detector.faster_rcnn).cuda()
    from models.detect_models.frcnn_utils.dataset import preprocess

##### class_dict ######
class_dict = ['person', 'car', 'bus', 'bicycle', 'motorcycle']

writer = SummaryWriter(opt.log_path)
####### Global Variables End #######


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
def align_to_four(img, factor=16):
    #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    #align to four
    a_row = int(img.shape[0] / factor) * factor
    a_col = int(img.shape[1] / factor) * factor
    img = img[0:a_row, 0:a_col]
    #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
    return img

def read_img(file):
    img = align_to_four(cv2.imread(file))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).unsqueeze(0).to(device).to(torch.float32)/255.
    return img.permute(0, 3, 1, 2)

def checkpoint(epoch):
    
    model_out_path = opjoin(opt.log_path, 'weight', "net_{}.pkl".format(epoch))

    torch.save({'state_dict' : model.state_dict()}, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))


def check_invalid_ann(ps, h, w):
    xmin, ymin, xmax, ymax = ps
    invalid = False
    if any(ps) < 0:
        invalid = True
    if xmin > xmax or ymin > ymax:
        invalid = True
    if xmin >= w or ymin >= h:
        invalid = True
    return invalid

def read_assigned_target(fpath, x_meta, h, w):
    with open(fpath, 'r') as f:
        lines = f.readlines()
    obj_arr = []
    y_min_shift, x_min_shift, _, _, scale = x_meta
    for line in lines:
        class_name, xmin, ymin, xmax, ymax = line.strip().split(' ')
        xmin, ymin, xmax, ymax = min(w, int(xmin)), min(h, int(ymin)), min(w, int(xmax)), min(h, int(ymax))
        if check_invalid_ann([xmin, ymin, xmax, ymax], h, w):
            print(fpath, "is invalid for", line.strip(), h, w, '!')
            continue
        xmax = min(xmax, w)
        ymax = min(ymax, h)

        obj_arr.append([class_dict.index(class_name), 
                        xmin*scale+x_min_shift, 
                        ymin*scale+y_min_shift, 
                        xmax*scale+x_min_shift, 
                        ymax*scale+y_min_shift, ])
    return np.array(obj_arr)

def get_attack(input_img, gt, an_file, eps=1/255., eps_iter=0.5/255., n_iter=2):
    h, w, _ = input_img.shape
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    if gt is not None:
        gt_query, _ = letterbox_image_padded(gt, size=detector.model_img_size)
    else:
        gt_query = None
    an = read_assigned_target(opjoin(opt.an_path, an_file), x_meta, h, w)    

    trainer.train_step(gt_query, an, reverse=False)
    trainer.train_step(x_query, an, reverse=True)
#    trainer.train_step(gt_query, an, reverse=False)  

    hq_ = tog_mislabeling(  victim=detector, 
                            x_query=gt_query, 
                            target='ass', 
                            n_iter=n_iter, 
                            eps=eps, 
                            eps_iter=eps_iter,
                            detections_target=an, 
                            target_confidence=1., 
                            x_meta=x_meta,
                            gt=None,#gt_query,
                            onlyGrad=False,
                        )
    y1, x1, y2, x2, _ = x_meta
    hq_ = hq_[0, y1:y2, x1:x2, ]
    hq_ = cv2.resize(hq_, (w, h), cv2.INTER_CUBIC)
    return hq_

def adjust_gt(gt, hq_, lq):
    # grad1 = torch.sign(hq_ - lq)
    # grad2 = torch.sign(gt - lq)
    # gt[grad1!=grad2] = hq_[grad1!=grad2]
    # return gt
    return hq_

def detect(input_img, ):
    h, w, _ = input_img.shape
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
    return visualize_detections_cv2((x_query, detections_query, detector.model_img_size, detector.classes, x_meta), h, w, )

def train(model, criterion, optimizer, step, lq, gt, an_file, nstep, epoch, tmp_id = 0):

    lq = lq.to(device)
    gt = gt.to(device)


    hq = model(lq)
    hq = torch.clamp(hq, min=0, max=1)
    step_loss = criterion(hq, gt)
    if step % 50 == 0:
        if True: #not os.path.exists("./%s/tmp_det%0.6d_%0.3d.jpg" % (opt.tmppath, tmp_id, 0)):
            save_img = detect((hq.detach()[0].permute(1,2,0).cpu().numpy()), )
            cv2.imwrite("./%s/tmp_det%0.6d_%0.3d.jpg" % (opt.tmppath, tmp_id, epoch), save_img[...,::-1])
            save_img = detect((gt.detach()[0].permute(1,2,0).cpu().numpy()), )
            cv2.imwrite("./%s/tmp_det%0.6d_%0.3d_c.jpg" % (opt.tmppath, tmp_id, 0), save_img[...,::-1])

    if True:#step_loss.item() <= 0.05:
        print("activate attack", step_loss.item())
        save_img = detect((gt.detach()[0].permute(1,2,0).cpu().numpy()), )
        cv2.imwrite("./%s/a/tmp_det%0.6d_%0.3d_b.jpg" % (opt.tmppath, tmp_id, epoch), save_img[...,::-1])
   
        hq_ = get_attack(hq.detach()[0].permute(1,2,0).cpu().numpy(), gt.detach()[0].permute(1,2,0).cpu().numpy(), an_file)

        save_img = detect((hq_), )
        cv2.imwrite("./%s/a/tmp_det%0.6d_%0.3d.jpg" % (opt.tmppath, tmp_id, epoch), save_img[...,::-1])

        hq_ = torch.Tensor(hq_).unsqueeze(0).permute(0,3,1,2).to(device)
        gt = adjust_gt(gt, hq_, lq)

        step_loss = criterion(hq, gt)

    optimizer.zero_grad()
    step_loss.backward()
#    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.0001, norm_type=2)
    optimizer.step()

    print("===>Step{} Part: Avg loss is :{:4f}".format(step, step_loss.item()))
    writer.add_scalar("Train Loss", step_loss.item(), step+(epoch-1)*nstep)

    return step_loss


model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.7)
opt.nEpochs   = training_settings['nEpochs']
opt.lr_decay  = training_settings['lr_decay']
print(opt)

lq_files = natsorted(os.listdir(opt.lq_path))
# gt_files = natsorted(os.listdir(opt.gt_path))
# an_files = natsorted(os.listdir(opt.an_path))
gt_files = [f_[:-9]+'.png' for f_ in lq_files]
an_files = [f_[:-9]+'.txt' for f_ in lq_files]


for epoch in range(opt.start_epoch, opt.nEpochs+1):
    trainer.reset_meters()
    tmp_id = 0
    psnr = 0

    nstep = len(lq_files)
    assert len(lq_files) == len(gt_files), "%d != %d" % (len(lq_files), len(gt_files))
    writer.add_scalar("Learning rate", optimizer.state_dict()['param_groups'][0]['lr'], epoch)
    for i, (lq_file, gt_file) in enumerate(zip(lq_files, gt_files)):
        #if i > 20:
        #    break
        an_file = an_files[i]
        print("## %06d:LQ File %s, GT File %s, AN File %s " % (i, lq_file, gt_file, an_file))
        lq = read_img(os.path.join(opt.lq_path, lq_file))
        gt = read_img(os.path.join(opt.gt_path, gt_file))
#        if lq.shape[2] > 1000 or lq.shape[3] > 1000:
#            continue
        step_loss = train(model, criterion, optimizer, i, lq, gt, an_file, nstep, epoch, tmp_id)
        tmp_id += 1
        psnr = psnr + step_loss
        
    if epoch % int(opt.nEpochs / 10) == 0:
        checkpoint(epoch)

    psnr = psnr / len(lq_files)
    writer.add_scalar("Epoch psnr", psnr, epoch)
    scheduler.step(psnr)
    lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
    writer.add_scalar("FRCNN lr", lr_, epoch)
    print("===>Epoch{} Complete: Avg loss is :{:4f}".format(epoch, psnr))
    with open(opjoin(opt.log_path, "log.txt"), "a+") as text_file:
        print("===>Epoch{} Complete: Avg loss is :{:4f}\n".format(epoch, psnr), file=text_file)
print(class_dict)
