from .target_utils import generate_attack_targets
import numpy as np

def tog_mislabeling(victim, x_query, x_meta, target, n_iter=10, eps=8/255., eps_iter=2/255., detections_target=None, target_confidence=1.0, gt=None, onlyGrad=False):
    detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    detections_target = generate_attack_targets(detections_query, confidence_threshold=victim.confidence_thresh_default, mode=target, detections_target=detections_target, target_confidence=target_confidence)

    if onlyGrad:
        g = victim.compute_object_mislabeling_gradient(x_query, detections=detections_target)
        return np.clip(x_query - g, 0.0, 1.0)

    # print(detections_target)
    y_min_s, x_min_s, y_max_s, x_max_s, _ = x_meta
    # print(y_min_s, x_min_s, y_max_s, x_max_s)

    if gt is not None:
        eta = np.sign(gt - x_query) * eps
    else:
        eta = np.random.uniform(-eps, eps, size=x_query.shape)

    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    x_adv[:,:y_min_s] = 0; x_adv[:,y_max_s:] = 0; x_adv[:,:, :x_min_s] = 0; x_adv[:,:, x_max_s:] = 0; 
    prev_loss = 1
    beta1 = 0.9
    beta2 = 0.999
    momentum = 0; vol = 0;
    prev_loss = 1.
    for t in range(1, 1+n_iter):
        grad, loss = victim.compute_object_mislabeling_gradient_and_loss(x_adv, detections=detections_target)
        momentum = beta1 * momentum + (1 - beta1) * grad
        vol = beta2 * vol + (1 - beta2) * grad**2    
        m_ = momentum / (1 - beta1**t)
        v_ = vol / (1 - beta2**t)
        grad = m_ / (v_ ** 0.5 +  1e-8)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        x_adv[:,:y_min_s] = 0; x_adv[:,y_max_s:] = 0; x_adv[:,:, :x_min_s] = 0; x_adv[:,:, x_max_s:] = 0;
        if abs(loss - prev_loss) / prev_loss < 1e-3:
            break
        prev_loss = loss
    return x_adv

