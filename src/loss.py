
import torch
import torch.nn.functional as F

def stable_log1pex(x):
    return torch.log(1 + torch.exp(-torch.abs(x))) - torch.minimum(x, torch.tensor(0).to('cuda'))

def fake_negative_weighted_loss(targets, outputs, params=None):
    z = targets["label"].reshape(-1)
    x = outputs["logits"].reshape(-1)
    z = z.float()
    p_no_grad = torch.sigmoid(x).detach()
    pos_loss = (1+p_no_grad)*stable_log1pex(x)
    neg_loss = -(1-p_no_grad)*(1+p_no_grad)*(-x-stable_log1pex(x))
    loss = torch.mean(pos_loss*z + neg_loss*(1-z))
    return {"loss": loss}

def delay_tn_dp_loss(targets, outputs, params=None):
    tn = outputs["tn_logits"].float()
    dp = outputs["dp_logits"].float()
    z = targets["label"].float()
    tn_label = torch.reshape(z[:, 0], (-1, 1))
    dp_label = torch.reshape(z[:, 1], (-1, 1))
    pos_label = torch.reshape(z[:, 2], (-1, 1))
    tn_mask = (1-pos_label)+dp_label
    tn_loss = torch.sum(F.binary_cross_entropy_with_logits(input=tn, \
        target=tn_label, reduction='none')*tn_mask) / torch.sum(tn_mask)
    dp_loss = F.binary_cross_entropy_with_logits(input=dp, target=dp_label)
    loss = tn_loss + dp_loss
    return {
        "loss": loss,
        "tn_loss": tn_loss,
        "dp_loss": dp_loss
    }

def cross_entropy_loss(targets, outputs, params=None):
    z = targets["label"]
    x = outputs["logits"]
    z = z.float().reshape(-1)
    x = x.reshape(-1)
    loss = F.binary_cross_entropy_with_logits(input=x, target=z)
    return {"loss": loss}

def fsiw_loss(targets, outputs, params=None):
    x = outputs["logits"]
    logits0 = outputs["logits0"].float().detach()
    logits1 = outputs["logits1"].float().detach()
    prob0 = torch.sigmoid(logits0)
    prob1 = torch.sigmoid(logits1)
    z = torch.reshape(targets["label"].float(), (-1, 1))

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)
    pos_weight = 1/(prob1+1e-8)
    neg_weight = prob0
    clf_loss = torch.mean(pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z))
    loss = clf_loss
    return {
        "loss": loss,
    }

def ddfm_loss(targets, outputs, params=None):
    x = outputs["logits"]
    tn_logits = outputs["tn_logits"]
    dp_logits = outputs["dp_logits"] # p(y) p(d > e)
    z = targets["label"]
    z = torch.reshape(z.float(), (-1, 1))
    dp_label = targets["delay_label"]
    d = torch.reshape(dp_label.float(), (-1, 1))
    
    p_no_grad = torch.sigmoid(x.detach())
    dist_prob = torch.sigmoid(tn_logits.detach())
    dp_prob = torch.sigmoid(dp_logits.detach())
    
    z1 = dist_prob
    z2 = 1 - p_no_grad + dp_prob

    pos_loss = stable_log1pex(x)
    neg_loss = x + stable_log1pex(x)
    loss1 = pos_loss # IP
    loss2 = (1-z1) * pos_loss + z1 * neg_loss # FN + RN
    loss3 = pos_loss # DP
    loss4 = (1-z2) * pos_loss + z2 * neg_loss # DN
    loss = torch.mean((1-d)*(z*(loss1) + (1-z)*(loss2)) + d*(z*loss3 + (1-z)*loss4))
    return {"loss": loss}

def get_loss_fn(name):
    if name == "cross_entropy_loss":
        return cross_entropy_loss
    elif name == "fake_negative_weighted_loss":
        return fake_negative_weighted_loss
    elif name == "tn_dp_pretraining_loss":
        return delay_tn_dp_loss
    elif name == "fsiw_loss":
        return fsiw_loss
    elif name == "ddfm_loss":
        return ddfm_loss
    else:
        raise NotImplementedError("{} loss does not implemented".format(name))

