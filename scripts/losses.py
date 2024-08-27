import torch
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy import stats
def spearman(y_pred, y_true):
    if np.var(y_pred) < 1e-6 or np.var(y_true) < 1e-6:
        return 0.0
    return spearmanr(y_pred, y_true)[0]


def compute_stat(sr):
    sr = np.asarray(sr)
    mean = np.mean(sr)
    std = np.std(sr)
    return mean, std


# Define the cosine similarity function
def cosine_similarity(seq1, seq2):
    return torch.cosine_similarity(seq1, seq2)

# Define the criterion function
def criterion(seq1, seq2, label):
    similarity = cosine_similarity(seq1, seq2)
    loss = ListMLE_loss(similarity, label)
    sim = np.asarray(similarity.cpu().detach().numpy())
    lab = np.asarray(label.cpu().detach().numpy())
    sr = spearman(sim,lab)
    return sr,loss

# Define the BT_loss function
def BT_loss(scores, golden_score):
    loss = torch.tensor(0.).cuda()
    for i in range(len(scores)):
        for j in range(i, len(scores)):
            if golden_score[i] > golden_score[j]:
                loss += torch.log(1 + torch.exp(scores[j] - scores[i]))
            else:
                loss += torch.log(1 + torch.exp(scores[i] - scores[j]))
    return loss
def ListMLE_loss(predicts, targets):
    ''' ListMLE loss '''
    if predicts.dim() == 1:
        predicts = predicts.unsqueeze(0)
    if targets.dim() == 1:
        targets = targets.unsqueeze(0)
    indices = targets.sort(descending=True, dim=-1).indices
    predicts = torch.gather(predicts, dim=1, index=indices)
    cumsums = predicts.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    loss = torch.log(cumsums + 1e-10) - predicts
    return loss.sum(dim=1).mean()

def KLloss(logits, logits_reg, seq, att_mask):

    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(seq.shape[0])

    loss = torch.tensor(0.)
    loss = loss.cuda()
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)
    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]


        seq_len = torch.sum(att_mask[i])

        reg = probs_reg_i[torch.arange(0, seq_len), seq[i, :seq_len]]
        pred = probs_i[torch.arange(0, seq_len), seq[i, :seq_len]]

        loss += creterion_reg(reg.log(), pred)
    return loss

def SupConHardLoss(model_emb, temp, n_pos):
    '''
    return the SupCon-Hard loss
    features:  
        model output embedding, dimension [bsz, n_all, out_dim], 
        where bsz is batchsize, 
        n_all is anchor, pos, neg (n_all = 1 + n_pos + n_neg)
        and out_dim is embedding dimension
    temp:
        temperature     
    n_pos:
        number of positive examples per anchor
    '''
    # l2 normalize every embedding
    features = F.normalize(model_emb, dim=-1, p=2)
    # features_T is [bsz, outdim, n_all], for performing batch dot product
    features_T = torch.transpose(features, 1, 2)
    # anchor is the first embedding 
    anchor = features[:, 0]
    # anchor is the first embedding 
    anchor_dot_features = torch.bmm(anchor.unsqueeze(1), features_T)/temp 
    # anchor_dot_features now [bsz, n_all], contains 
    anchor_dot_features = anchor_dot_features.squeeze(1)
    # deduct by max logits, which will be 1/temp since features are L2 normalized 
    logits = anchor_dot_features - 1/temp
    # the exp(z_i dot z_a) excludes the dot product between itself
    # exp_logits is of size [bsz, n_pos+n_neg]
    exp_logits = torch.exp(logits[:, 1:])
    exp_logits_sum = n_pos * torch.log(exp_logits.sum(1)) # size [bsz], scale by n_pos
    pos_logits_sum = logits[:, 1:n_pos+1].sum(1) #sum over all (anchor dot pos)
    log_prob = (pos_logits_sum - exp_logits_sum)/n_pos
    loss = - log_prob.mean()
    return loss    
