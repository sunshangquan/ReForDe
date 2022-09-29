rm tmp2/* -r; python train.py --gpu_ids 0 --model_det yolov3 --model_res GridDehaze --lr 1e-7 --tmppath tmp2 --resume ./log/GridDehaze_yolov3_ft/weight/net_1.pkl
