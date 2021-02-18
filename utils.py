import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy
import itertools
import math

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def loss_adv(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def cosine_matrix(x,y):
    x=F.normalize(x,dim=1)
    y=F.normalize(y,dim=1)
    xty=torch.sum(x.unsqueeze(1)*y.unsqueeze(0),2)
    return 1-xty

def SM(Xs,Xt,Ys,Yt,Cs_memory,Ct_memory,Wt=None,decay=0.3):
    Cs=Cs_memory.clone()
    Ct=Ct_memory.clone()
    r = torch.norm(Xs,dim=1)[0]
    Ct=r*Ct/(torch.norm(Ct,dim=1,keepdim=True)+1e-10)
    Cs=r*Cs/(torch.norm(Cs,dim=1,keepdim=True)+1e-10)
    K=Cs.size(0)
    for k in range(K):
        Xs_k=Xs[Ys==k]
        Xt_k=Xt[Yt==k]
        if len(Xs_k)==0:
            Cs_k=0.0
        else:
            Cs_k=torch.mean(Xs_k,dim=0)

        if len(Xt_k)==0:
            Ct_k=0.0
        else:
            if Wt is None:
                Ct_k=torch.mean(Xt_k,dim=0)
            else:
                Wt_k=Wt[Yt==k]
                Ct_k=torch.sum(Wt_k.view(-1,1)*Xt_k,dim=0)/(torch.sum(Wt_k)+1e-5)
        Cs[k,:]=(1-decay)*Cs_memory[k,:]+decay*Cs_k
        Ct[k,:]=(1-decay)*Ct_memory[k,:]+decay*Ct_k
    Dist=cosine_matrix(Cs,Ct)
    return torch.sum(torch.diag(Dist)),Cs,Ct


def robust_pseudo_loss(output,label,weight,q=1.0):
    weight[weight<0.5] = 0.0
    one_hot_label=torch.zeros(output.size()).scatter_(1,label.cpu().view(-1,1),1).cuda()
    mask=torch.eq(one_hot_label,1.0)
    output=F.softmax(output,dim=1)
    mae=(1.0-torch.masked_select(output,mask)**q)/q
    return torch.sum(weight*mae)/(torch.sum(weight)+1e-10)

def to_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def to_onehot(label, num_classes):
    identity = to_cuda(torch.eye(num_classes))
    onehot = torch.index_select(identity, 0, label)
    return onehot

def cross_entropy_with_logits(preds, labels, label_smoothing=0.0, weight=None):
    if label_smoothing != 0:
        one_hot = labels.clone()
        labels -= label_smoothing * one_hot

        labels += (1 - one_hot) * (label_smoothing / labels.size(1))
    labels = labels.detach()

    if weight is None:
        loss = torch.mean(torch.sum(- labels*F.log_softmax(preds, dim=1), dim=1))
    else:
        loss = torch.sum(weight*torch.sum(- labels*F.log_softmax(preds, dim=1), dim=1)) / (torch.sum(weight)+1e-6)
    return loss

def mosaic(x, y, num_patch_1, num_patch_2, with_mixup=False, with_random=False, with_mosaic_v1=False, with_mosaic_v2=False):
    
    size = x.size()
    
    new_x = deepcopy(x)

    if with_mosaic_v1:
        x_ = []
        for i in range(num_patch_1):
            for j in range(num_patch_1):
                x_.append(new_x[:, :, i::num_patch_1, j::num_patch_1])
        x_p = torch.stack(x_, dim=0)

        indx0 = []
        for i in range(size[0]):
            indx0.append(torch.randperm(num_patch_1*num_patch_1))
        indx0 = torch.stack(indx0, dim=1)
        indx1 = torch.arange(size[0])

        for i in range(num_patch_1):
            for j in range(num_patch_1):
                new_x[:, :, i::num_patch_1, j::num_patch_1] = x_p[indx0[i*num_patch_1+j], indx1]

    if with_mosaic_v2:
        x_ = []
        expect = []
        patch_size = size[2] // num_patch_2

        for i in range(num_patch_2):
            for j in range(num_patch_2):
                if (i*num_patch_2+j) not in expect:
                    patch = new_x[:, :, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)]
                    if random.random() < 0.5:
                        patch = torch.flip(patch, dims=[3])
                    x_.append(patch)

        x_p = torch.stack(x_, dim=0)
        
        indx0 = []
        for i in range(size[0]):
            indx0.append(torch.randperm(num_patch_2*num_patch_2))
        indx0 = torch.stack(indx0, dim=1)
        indx1 = torch.arange(size[0])

        k = 0
        for i in range(num_patch_2):
            for j in range(num_patch_2):
                if (k) not in expect:
                    new_x[:, :, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)] = x_p[indx0[k], indx1]
                    k += 1

    if with_random:
        new_x += (torch.randn_like(new_x) - 0.5) / 4
        # lam = lam.view([-1, 1])
        # y -= lam * y
    
    if with_mixup:
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
        
        lam = np.random.beta(0.5, 0.5, size=(y.size(0), 1, 1, 1))

        lam = torch.from_numpy(lam).to(torch.float).cuda()
        batch_size = x.size()[0]
        
        index = torch.randperm(batch_size).cuda()

        new_x = lam * new_x + (1. - lam) * new_x[index]
        lam = lam.view([-1, 1])
        y = lam * y + (1. - lam) * y[index]

    return new_x.detach(), y.detach()

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False)
        # self.criterion = cross_entropy_with_logits

        self.matrix = torch.zeros((batch_size*2, batch_size*2)).cuda()
        index_1 = np.arange(batch_size*2)
        index_2 = [(x+batch_size)%(batch_size*2) for x in index_1]
        self.matrix[index_1, index_2] = 1
        self.matrix[index_1, index_1] = -1

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs, ilabels, jlabels, weights, idx, normalize=True, class_level=False):
        if normalize:
            zjs = F.normalize(zjs,dim=1)
            zis = F.normalize(zis,dim=1)
        
        # representations = torch.cat([zjs, zis], dim=0)

        # similarity_matrix = self.similarity_function(representations, representations)
        # similarity_matrix = self.similarity_function(zjs, zis)
        similarity_matrix = 1. - cosine_matrix(zjs, zis)
        labels = (ilabels.unsqueeze(0) == jlabels.unsqueeze(1)).float()

        # labels = torch.cat([jlabels, ilabels], dim=0)
        # labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels / (labels.sum(1, keepdim=True) + 1e-8)

        logits = similarity_matrix / self.temperature
        # logits = (similarity_matrix - 0.5) / 0.3
        # logits = (similarity_matrix - 0.5) / 0.5

        # # loss = cross_entropy_with_logits(logits, labels)
        # pos_w = torch.ones([1]).cuda()
        # weight = np.array(weights)
        # weight = torch.from_numpy(weight).view(1, -1).cuda().float()
        # loss = nn.BCEWithLogitsLoss(pos_weight=pos_w, reduce=True)(logits, labels)
        # # loss *= weight
        # # loss = (loss.sum(dim=1) / weight.sum(1)).mean()

        # return loss

        # loss = torch.sum((1. - similarity_matrix) * labels) / torch.sum(labels)
        
        
        losses = 0
        for i in range(labels.size(0)):

            zeros = torch.nonzero(labels[i] - 1.0).view(-1)
            if class_level:
                ones = torch.nonzero(labels[i]).view(-1)
                size = ones.size(0)
            else:
                ones = torch.from_numpy(np.array(idx + i)).cuda()
                size = 1

            label = torch.zeros(size).cuda().long()
            if size == 0 or size == labels.size(1):
                continue
            weight = np.array(weights[ones])
            weight = torch.from_numpy(weight).view(-1).cuda().float()

            zeros = zeros.repeat(size, 1).view(-1)
            neg = torch.gather(logits[i], 0, zeros).view(size, -1)
            pos = torch.gather(logits[i], 0, ones).view(size, 1)
            logits_ = torch.cat([pos, neg], dim=1)
            loss = self.criterion(logits_, label)
            loss *= weight
            losses += loss.sum() / weight.sum()
        return losses / labels.size(0)


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.99, updates=0):
        # Create EMA
        self.ema = deepcopy(model).eval()  # FP32 EMA
        # if next(model.parameters()).device.type != 'cpu':
        #     self.ema.half()  # FP16 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 400))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            # d = 0.999

            msd =  model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def rightshift(x, k):
    temp = deepcopy(x)
    x[k:] = temp[:-k]
    x[:k] = temp[-k:]

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 