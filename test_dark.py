from __future__ import print_function
import argparse
import os, cv2, sys
import time
from math import log10
from torchvision import transforms
from torchvision import utils as utils
import torch
from importlib import import_module
import statistics
import re
import torch.nn as nn
import torch.nn.functional as F
import utils.pytorch_ssim as pytorch_ssim
from natsort import natsorted
from PIL import Image
from collections import OrderedDict


sys.path.append('/home1/ssq/proj3/reforde2/models/detect_models/')
from utils.dataset_utils.preprocessing import letterbox_image_padded, letterbox_image_padded_PIL
from utils.misc_utils.visualization import visualize_detections, visualize_detections_cv2

parser = argparse.ArgumentParser(description="PyTorch Test")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")

parser.add_argument('--lq_path', type=str, default='../VOC_dark/test/data', help='Path of the validation dataset')
parser.add_argument('--gt_path', type=str, default='../VOC_dark/test/gt', help='Path of the validation dataset')
parser.add_argument('--save_path', type=str, default='./output2/', help='Path of the validation dataset')

parser.add_argument("--checkpoint", default="pretrained_weights/restore_models/MSBDN/ft_1.pkl", type=str, help="Test on intermediate pkl (default: none)")
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--model_res', type=str, default='MSBDN', help='filename of the training models')
parser.add_argument('--task', type=str, default='dehaze', help='filename of the training models')

parser.add_argument('--model_det', type=str, default='yolov3', help='filename of the training models')

parser.add_argument('--det_size', type=int, default=640, help='')
parser.add_argument("--ifnotRes", action='store_true')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####### Global Variables #######
opt = parser.parse_args()
assert opt.model_res.lower() in ['zerodce', 'sci']
opt.save_path = os.path.join(opt.save_path, opt.model_res, opt.model_det+'_nores_exdark_544')#"_attlr8nomodelnet3_voc_dark_m")
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids[0]
if opt.gt_path != '':
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
else:
    criterion = None

device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')

if opt.model_res.lower() == 'zerodce':
    ZeroDCE = import_module("models.restore_models.ZeroDCE.model")

    model = ZeroDCE.enhance_net_nopool()
    try:
        model.load_state_dict(torch.load(opt.checkpoint, map_location=device)['state_dict'], strict=True)
    except:
        model.load_state_dict(torch.load(opt.checkpoint), strict=True)
elif opt.model_res.lower() == 'sci':
    SCI = import_module("models.restore_models.SCI.model")
    model = SCI.Finetunemodel(opt.checkpoint).to(device)
    '''
    try:
        model.load_state_dict(torch.load(opt.checkpoint)['state_dict'], strict=True)
    except:
        checkpoint_ = torch.load(opt.checkpoint)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint_.items():
            name = k[7:] if 'module.' in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
    '''

model = model.eval().to(device)
if opt.model_det.lower() == 'yolov3':
    import tensorflow as tf
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess = tf.Session(config=config)
    yolov3 = import_module("models.detect_models." + opt.model_det.lower())
    detector = yolov3.YOLOv3_Darknet53_10(weights='pretrained_weights/detect_models/yolov3_10_worst.h5', model_img_size=(opt.det_size, opt.det_size))
elif opt.model_det.lower() == 'frcnn':
    frcnn = import_module("models.detect_models." + opt.model_det.lower())
    detector = frcnn.FRCNN(model_img_size=(opt.det_size, opt.det_size), device=device, dark=True).to(device).load('pretrained_weights/detect_models/FRCNN_10_best.pth')
print(opt)
####### Global Variables #######

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

def detect(input_img, save_path):
    h, w, _ = input_img.shape
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    detections_query = detector.detect(x_query, conf_threshold=detector.confidence_thresh_default)
    return visualize_detections_cv2((x_query, detections_query, detector.model_img_size, detector.classes, x_meta), h, w, save_path)

def test(model, lq_path, gt_path, save_path):
    avg_psnr = 0
    avg_ssim = 0
    med_time = []
    psnr = 0
    lq_files = natsorted(os.listdir(lq_path))
    os.makedirs(os.path.join(save_path, 'output_res'), exist_ok=True)

    gt_files = natsorted(os.listdir(gt_path)) if gt_path != '' else []
    assert len(lq_files) == len(gt_files) or len(gt_files) == 0, "%d != %d" % (len(lq_files), len(gt_files))
    with torch.no_grad():
        for i, lq_file in enumerate(lq_files):
            if os.path.exists(os.path.join(save_path, 'output_res', lq_file)):
                print('skip', lq_file)
                continue
            lq = read_img(os.path.join(lq_path, lq_file))
            
            start_time = time.perf_counter()

            hq = lq if opt.ifnotRes else model(lq)
            hq = torch.clamp(hq, min=0, max=1)
            
            evalation_time = time.perf_counter() - start_time
            med_time.append(evalation_time)

            if gt_path != '': 
                gt = read_img(os.path.join(gt_path, gt_files[i]))
                ssim = pytorch_ssim.ssim(hq, gt)
                #print(ssim)
                avg_ssim += ssim
                mse = criterion(hq, gt)
                psnr = 10 * log10(1 / mse)
                
                avg_psnr += psnr
            print("Processing %06d:  PSNR:%.2f TIME:%.4f, %s" % (i, psnr, evalation_time, lq_file))
            save_img = hq.cpu()[0].permute(1,2,0).numpy()
            tmp = lq_file.split('.')[0]
            save_img = detect(save_img, os.path.join(save_path, 'output_det', tmp+".txt"),)

            save_img = Image.fromarray(save_img).convert('RGB')
            save_img.save(os.path.join(save_path, 'output_res', lq_file))


        print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / i))
        print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / i))
        median_time = statistics.median(med_time)
        print(median_time)
           
    
def inference():
    str_ids = opt.gpu_ids.split(',')
    torch.cuda.set_device(int(str_ids[0]))
    lq_path = opt.lq_path# #----------Validation path
    gt_path = opt.gt_path# #----------Validation path
    save_path = opt.save_path  #--------------------------SR results save path
    isexists = os.path.exists(save_path)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'output_det'), exist_ok=True)
    print("The results of testing images sotre in {}.".format(save_path))

    print("===> Loading model")

    print("Testing model {}----------------------------------".format(opt.checkpoint))
    print(get_n_params(model))
    #model = model.eval()
    test(model, lq_path, gt_path, save_path)

if __name__ == '__main__':
    inference()

