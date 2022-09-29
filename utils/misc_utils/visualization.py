import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_detections(detections_dict, save_path):
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.clf()
    plt.figure(figsize=(8 * len(detections_dict), 8))
    for pid, title in enumerate(detections_dict.keys()):
        input_img, detections, model_img_size, classes, meta = detections_dict[title]
        y1, x1, y2, x2, _ = meta
        if len(input_img.shape) == 4:
            input_img = input_img[0]
        plt.subplot(1, len(detections_dict), pid + 1)
        plt.title(title)
        plt.imshow(input_img[y1:y2, x1:x2])
        current_axis = plt.gca()
        for box in detections:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1])-x1, 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0])-y1, 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1])-x1, input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0])-y1, input_img.shape[0])
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='small', color='black', bbox={'facecolor': color, 'alpha': 1.0})
        plt.axis('off')
    # plt.show()
    print(save_path)
    plt.savefig(save_path)

def visualize_detections_cv2(detections, h, w, save_path=None, an=None):
    colors = plt.cm.hsv(np.linspace(0, 1, 11)).tolist()
    input_img, detections, model_img_size, classes, meta = detections
    y1, x1, y2, x2, _ = meta
    if len(input_img.shape) == 4:
        input_img = input_img[0]
    save_img = input_img[y1:y2, x1:x2]*255.
    save_img = save_img.astype("uint8")
    if save_path:
        f = open(save_path, 'a+')
    for box in detections:
        xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1])-x1, 0)
        ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0])-y1, 0)
        xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1])-x1, input_img.shape[1])
        ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0])-y1, input_img.shape[0])
        color = [i*255 for i in colors[int(box[0])]]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        cv2.rectangle(save_img, (xmin, ymin), (xmax, ymax), color=color[:3], thickness=4)
        cv2.putText(save_img, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        if save_path:
            f.write("{} {} {} {} {} {}\n".format(classes[int(box[0])], box[1], int(xmin/save_img.shape[1]*w), int(ymin/save_img.shape[0]*h), int(xmax/save_img.shape[1]*w), int(ymax/save_img.shape[0]*h)))
    if an is not None:
        for box in an:
            xmin = max(int(box[-4] * input_img.shape[1] / model_img_size[1])-x1, 0)
            ymin = max(int(box[-3] * input_img.shape[0] / model_img_size[0])-y1, 0)
            xmax = min(int(box[-2] * input_img.shape[1] / model_img_size[1])-x1, input_img.shape[1])
            ymax = min(int(box[-1] * input_img.shape[0] / model_img_size[0])-y1, input_img.shape[0])
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])

            cv2.rectangle(save_img, (xmin, ymin), (xmax, ymax), color=(0,0,0), thickness=5)
            cv2.putText(save_img, label, (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 3)
    save_img = cv2.resize(save_img, (w, h), cv2.INTER_CUBIC)
    return save_img
