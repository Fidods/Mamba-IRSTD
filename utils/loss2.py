import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
import cv2

def FocalIoULoss(inputs, targets):
    "Non weighted version of Focal Loss"

    # def __init__(self, alpha=.25, gamma=2):
    #     super(WeightedFocalLoss, self).__init__()
    # targets =
    # inputs = torch.relu(inputs)
    [b,c,h,w] = inputs.size()

    inputs = torch.nn.Sigmoid()(inputs)
    inputs = 0.999*(inputs-0.5)+0.5
    BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
    intersection = torch.mul(inputs, targets)
    smooth = 1

    IoU = (intersection.sum() + smooth) / (inputs.sum() + targets.sum() - intersection.sum() + smooth)
    
    alpha = 0.75
    gamma = 2
    num_classes = 2
    # alpha_f = torch.tensor([alpha, 1 - alpha]).cuda()
    # alpha_f = torch.tensor([alpha, 1 - alpha])
    gamma = gamma
    size_average = True

 
    pt = torch.exp(-BCE_loss)
    
    F_loss =  torch.mul(((1-pt) ** gamma) ,BCE_loss)
   
    at = targets*alpha+(1-targets)*(1-alpha)
    
    F_loss = (1-IoU)*(F_loss)**(IoU*0.5+0.5)
   
    F_loss_map = at * F_loss


    F_loss_sum = F_loss_map.sum()
    
    return F_loss_sum
#   returm F_loss_map, F_loss_sum



def SoftIoULoss(pred, target):
        pred = torch.nn.Sigmoid()(pred)
       

        intersection = torch.mul(pred,target)

       
        smooth = 1

        loss = (intersection.sum() +smooth) / (pred.sum() + target.sum() -intersection.sum() + smooth)
       
        loss = 1 - loss.mean()

        return loss


class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()

    def forward(self, pred_log, target, warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        smooth = 0.0

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        dis = torch.pow((pred_sum - target_sum) / 2, 2)

        alpha = (torch.min(pred_sum, target_sum) + dis + smooth) / (torch.max(pred_sum, target_sum) + dis + smooth)

        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)
        lloss = LLoss(pred, target)

        if epoch > warm_epoch:
            siou_loss = alpha * loss
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 - siou_loss.mean()
        else:
            loss = 1 - loss.mean()
        return loss


def LLoss(pred, target):
    loss = torch.tensor(0.0, requires_grad=True).to(pred)

    patch_size = pred.shape[0]
    h = pred.shape[2]
    w = pred.shape[3]
    x_index = torch.arange(0, w, 1).view(1, 1, w).repeat((1, h, 1)).to(pred) / w
    y_index = torch.arange(0, h, 1).view(1, h, 1).repeat((1, 1, w)).to(pred) / h
    smooth = 1e-8
    for i in range(patch_size):
        pred_centerx = (x_index * pred[i]).mean()
        pred_centery = (y_index * pred[i]).mean()

        target_centerx = (x_index * target[i]).mean()
        target_centery = (y_index * target[i]).mean()

        angle_loss = (4 / (torch.pi ** 2)) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth))
                                                           - torch.arctan(
            (target_centery) / (target_centerx + smooth))))

        pred_length = torch.sqrt(pred_centerx * pred_centerx + pred_centery * pred_centery + smooth)
        target_length = torch.sqrt(target_centerx * target_centerx + target_centery * target_centery + smooth)

        length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)

        loss = loss + (1 - length_loss + angle_loss) / patch_size

    return loss
def FocalLoss(inputs, targets):

    alpha = 0.75
    gamma = 2
    num_classes = 2

  
    gamma = gamma
    size_average = True


    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    targets = targets.type(torch.long)

    
    at = targets*alpha+(1-targets)*(1-alpha)
    pt = torch.exp(-BCE_loss)
    F_loss = (1 - pt) ** gamma * BCE_loss

    F_loss = at * F_loss
    return F_loss.sum()



