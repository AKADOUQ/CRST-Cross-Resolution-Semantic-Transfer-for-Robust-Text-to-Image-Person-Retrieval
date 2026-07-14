import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sdm(image_fetures, text_fetures, pid, logit_scale, image_id=None, factor=0.3, epsilon=1e-8):
    batch_size = image_fetures.shape[0]
    pid = pid.reshape((batch_size, 1)) 
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    if image_id != None:
        image_id = image_id.reshape((-1, 1))
        image_id_dist = image_id - image_id.t()
        image_id_mask = (image_id_dist == 0).float()
        labels = (labels - image_id_mask) * factor + image_id_mask

    image_norm = image_fetures / image_fetures.norm(dim=1, keepdim=True)
    text_norm = text_fetures / text_fetures.norm(dim=1, keepdim=True)

    t2i_cosine_theta = text_norm @ image_norm.t()
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss


def compute_mlm(scores, labels):
    ce = nn.CrossEntropyLoss(ignore_index=0)
    return ce(scores, labels)


def compute_itc(image_features, text_features, logit_scale):
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t =F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss


def compute_id(image_logits, text_logits, labels):
    criterion = nn.CrossEntropyLoss(reduction="mean")

    loss = criterion(image_logits, labels) + criterion(text_logits, labels)
    
    return loss / 2


def compute_cmpm(image_embeddings, text_embeddings, labels, epsilon=1e-8):
    batch_size = image_embeddings.shape[0]
    labels_reshape = torch.reshape(labels, (batch_size, 1))
    labels_dist = labels_reshape - labels_reshape.t()
    labels_mask = (labels_dist == 0).float()

    image_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
    text_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
    image_proj_text = torch.matmul(image_embeddings, text_norm.t())
    text_proj_image = torch.matmul(text_embeddings, image_norm.t())

    # normalize the true matching distribution
    labels_mask_norm = labels_mask / labels_mask.norm(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_mask_norm + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_mask_norm + epsilon))

    cmpm_loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return cmpm_loss

def compute_cr_sdm(image_logits_hr, image_logits_lr, text_logits_hr, text_logits_lr):
    p_i2t_hr = F.softmax(image_logits_hr, dim=1)
    p_i2t_lr = F.log_softmax(image_logits_lr, dim=1)
    kl_i2t = F.kl_div(p_i2t_lr, p_i2t_hr, reduction='batchmean')
    p_t2i_hr = F.softmax(text_logits_hr, dim=1)
    p_t2i_lr = F.log_softmax(text_logits_lr, dim=1)
    
    kl_t2i = F.kl_div(p_t2i_lr, p_t2i_hr, reduction='batchmean')
    
    return (kl_i2t + kl_t2i) / 2

def compute_sr_loss(feat_student, feat_teacher):
    return compute_feature_consistency(feat_student, feat_teacher)


def _cosine_logits(image_features, text_features, logit_scale):
    image_norm = F.normalize(image_features, p=2, dim=-1)
    text_norm = F.normalize(text_features, p=2, dim=-1)
    return logit_scale * image_norm @ text_norm.t()


def compute_cr_rda(image_features_hr, image_features_lr, text_features, logit_scale):
    logits_i2t_hr = _cosine_logits(image_features_hr.detach(), text_features, logit_scale)
    logits_i2t_lr = _cosine_logits(image_features_lr, text_features, logit_scale)

    target_i2t = F.softmax(logits_i2t_hr.detach(), dim=1)
    log_prob_i2t = F.log_softmax(logits_i2t_lr, dim=1)
    loss_i2t = F.kl_div(log_prob_i2t, target_i2t, reduction='batchmean')

    logits_t2i_hr = logits_i2t_hr.t()
    logits_t2i_lr = logits_i2t_lr.t()
    target_t2i = F.softmax(logits_t2i_hr.detach(), dim=1)
    log_prob_t2i = F.log_softmax(logits_t2i_lr, dim=1)
    loss_t2i = F.kl_div(log_prob_t2i, target_t2i, reduction='batchmean')
    return loss_i2t + loss_t2i

def compute_feature_consistency(feat_student, feat_teacher):
    return F.mse_loss(feat_student, feat_teacher.detach())
