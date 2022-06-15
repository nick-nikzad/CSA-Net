# -*- coding: utf-8 -*-
"""
Created on Sat May 16 17:57:11 2020

@author: Nick-Nikzad
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class CSAtt(nn.Module):
    def __init__(self, channel=64, reduction_ratio=16):
        super(CSAtt, self).__init__()
        rdim = dict([(64,35), (128,28), (256,14) ,(512,7),(1024,4),(2048,4)])
        self.adapt_pool = nn.AdaptiveAvgPool2d(rdim[channel])
        
        self.gate_channels = channel
        self.mlp_D = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio)
            )
        self.non_linear= nn.ReLU(inplace=True)
        self.mlp_U = nn.Linear(channel // reduction_ratio, channel)
 

    def MoransI(self,x,ch_disc=None):
        device = x.device

        (batch_size, CH,h_i, w_i) = tuple(x.size())
        shrink_fmaps = self.adapt_pool(x)

        x_ap = torch.reshape(shrink_fmaps,(batch_size, CH,-1))
        
        if ch_disc is None:
            x_gap = F.adaptive_avg_pool2d(x,(1,1))#
        else:
            x_gap = ch_disc
            
        squeeze_ch_=torch.reshape(x_gap,(-1,CH,1))
        
        Wij=batch_spatial_weight(x_ap.to(device),x_ap.to(device),mode='hybrid')# 
        
    
    
        ######################################## Local Moran's index
        
        squeeze_ch_=torch.reshape(squeeze_ch_,(batch_size,CH,1)) # b * c * 1
      
        x_std=squeeze_ch_.std(dim=1,keepdim=True)
        x_mu=squeeze_ch_.mean(dim=1,keepdim=True)
        z_values=(squeeze_ch_-x_mu)/(x_std)        
        z_t=z_values.transpose(1,2)
        

        zxz_t=torch.bmm(z_values,z_t)
        M_star=torch.bmm(zxz_t,Wij) # b * c * c
     
        
        Local_MI=(torch.reshape(M_star.diagonal(dim1=1,dim2=2),(batch_size,CH,1))) # b * c * 1
        Local_MI = normal_G(Local_MI)

        
        return Local_MI.unsqueeze(-1)# b * c * 1 * 1

    
    def forward(self, x, ch_disc=None):
        ch_csa = self.MoransI(x, ch_disc)
        channel_att_raw = self.mlp_D(ch_csa)
        channel_att_raw=self.non_linear(channel_att_raw)
        channel_att_raw= self.mlp_U(channel_att_raw)
        scale = torch.sigmoid(channel_att_raw).unsqueeze(2).unsqueeze(3)
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def normal_G(tensor):
    epsilon=1e-12
    t_m = tensor.mean(dim=1,keepdim=True)
    t_std=tensor.std(dim=1,keepdim=True)
    tensor_norm = ((tensor - t_m) / (t_std+epsilon))
    return tensor_norm

  
def l2_pairdist(x,y):
    with torch.no_grad():
        x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
        y_t = y.permute(0,2,1).contiguous()
        y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
        # dist=torch.cdist(x, y, p=2)
        dist = torch.sqrt(dist)#dist#
        dist[dist != dist] = 0 # replace nan values with 0
    
        dist=torch.clamp(dist, 0.0, np.inf) 
    return dist

 
def pairwise_sim(x, y,mode='hybrid'):
  '''                                                                                              
  Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
  Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
  Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
  i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
  '''                                                                                
  epsilon=1e-10
  device = x.device
  x=x.to(device)
  y=y.to(device)
  (batch_size, CH,N) = tuple(x.size())
  
  if mode=='l2':
      x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
      y_t = y.permute(0,2,1).contiguous()
      y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
      dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
      # dist=torch.cdist(x, y, p=2)
      dist = torch.sqrt(dist)#dist#
      dist[dist != dist] = 0 # replace nan values with 0

      dist=torch.clamp(dist, 0.0, np.inf)      
      mean_dist=dist.mean(dim=(1,2),keepdim=True)
      dist_sim=torch.exp(-1*torch.div(dist,mean_dist+epsilon)).type(torch.float32).to('cuda')
      
  elif mode=='cosine':  
      w1 = x.norm(p=2, dim=2, keepdim=True)
      w2 = y.norm(p=2, dim=2, keepdim=True)
      y_t=y.transpose(1,2)
      w2_t=w2.transpose(1,2)
      dist_sim = torch.bmm(x, y_t)/((w1*w2_t)).clamp(min=epsilon)
      dist_sim[dist_sim != dist_sim] = 0 # replace nan values with 0

  elif mode=='l1':
      dist=torch.cdist(x, y, p=1)      
      dist[dist != dist] = 0 # replace nan values with 0
      dist=torch.clamp(dist, 0.0, np.inf)  
      mean_dist=dist.mean(dim=(1,2),keepdim=True)
      dist_sim=torch.exp(-1*torch.div(dist,mean_dist+epsilon)).type(torch.float32).to('cuda')      

  elif mode=='hybrid':
      
      ######################   L2 dist      
      x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
      y_t = y.permute(0,2,1).contiguous()
      y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
      L2_dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
      # dist=torch.cdist(x, y, p=2)
      L2_dist = torch.sqrt(L2_dist)#dist#
      L2_dist[L2_dist != L2_dist] = 0 # replace nan values with 0
      L2_dist=torch.clamp(L2_dist, 0.0, np.inf)         
      mean_dist=L2_dist.mean(dim=(1,2),keepdim=True)
      L2_sim=torch.exp(-1*torch.div(L2_dist,mean_dist+1e-10)).type(torch.float32).to('cuda')
      ######################### Cosine
      w1 = x.norm(p=2, dim=2, keepdim=True)
      w2 = y.norm(p=2, dim=2, keepdim=True)
      y_t=y.transpose(1,2)
      w2_t=w2.transpose(1,2)
      cos_sim = torch.bmm(x, y_t)/((w1*w2_t)).clamp(min=epsilon)
      cos_sim[cos_sim != cos_sim] = 0 # replace nan values with 0
      sim_cosine=torch.clamp(cos_sim, 0.0, np.inf)
      
      dist_sim=sim_cosine*L2_sim
      
      
    
  return dist_sim 
def batch_spatial_weight(x, y,mode='hybrid'):
  '''                                                                                              
  Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
  Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
  Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
  i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
  '''                                                                                
  device = x.device
  with torch.no_grad():
      epsilon=1e-10
      
      x=x.to(device)
      y=y.to(device)
      (batch_size, CH,N) = tuple(x.size())
      sp_w = pairwise_sim(x, y,mode=mode)
      I=torch.eye(sp_w.shape[1])
      Ix=torch.unsqueeze(I,0)
      

      one_mask=torch.ones_like(sp_w)
      diag_mask=one_mask.to(device)-Ix.to(device)
        
      Vij=sp_w*diag_mask
      Wij=Vij/(Vij.sum((1,2),keepdim=True)+epsilon)
  
  return Wij

