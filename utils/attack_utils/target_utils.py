from scipy.special import softmax
import numpy as np


def generate_attack_targets(detections, mode, confidence_threshold, class_id=None, detections_target=None, target_confidence=1.0):
    assert mode.lower() in ['ml', 'll', 'ass'], '`mode` should be one of `ML` or `LL` or `ass`.'
    detections_copy = detections.copy()
    pred_logits = detections_copy[:, 2:-4]

    if mode.lower() == 'ass':
        _, T = detections.shape
        m, _ = detections_target.shape
        detections_target_return = target_confidence * np.ones((m, T), dtype=detections.dtype)
        detections_target_return[:, 0] = detections_target[:, 0]
        detections_target_return[:, -4:] = detections_target[:, -4:]
#        try:
#            detections_target_return[:, 2:-4] = pred_logits[:m,]
#        except:
        pred_logits = 5. * np.ones((m, T-6), dtype=detections.dtype)
        detections_target_return[:, 2:-4] = pred_logits
        return detections_target_return
    if mode.lower() == 'll':
        if pred_logits.shape[1] % 10 == 1:  # ignore index 1 if it is referring to background class (SSD and FRCNN)
            pred_logits[:, 0] = float('inf')
        target_class_id = np.expand_dims(np.argmin(pred_logits, axis=-1), axis=1)
    else:
        pred_logits[softmax(pred_logits, axis=-1) > confidence_threshold] = float('-inf')
        if pred_logits.shape[1] % 10 == 1:  # ignore index 1 if it is referring to background class (SSD and FRCNN)
            pred_logits[:, 0] = float('-inf')
        target_class_id = np.expand_dims(np.argmax(pred_logits, axis=-1), axis=1)
    
    if class_id is not None:
        if pred_logits.shape[1] % 10 == 1:  # account for the background class in SSD and FRCNN
            class_id += 1
        source_class_id = detections_copy[:, [0]]
        mask = detections_copy[:, [0]] == class_id
        if np.sum(mask) == 0:
            return None
        target_class_id = np.where(mask, target_class_id, source_class_id)

    target_conf = np.full_like(target_class_id, fill_value=1.)
    detections_target = np.concatenate([target_class_id, target_conf, detections[:, 2:]], axis=-1)
    return detections_target
