# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:32:05 2024

@author: talha
"""
#%%
from torch import nn
from torch import cat
import torch
import torch.nn.functional as F
import time
import torch.nn.init as init
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
###MFI block
class mod_diff_code_block(nn.Module):
    def __init__(self,in_channel, num_classes=5):
        super(mod_diff_code_block,self).__init__()
        self.Relation1 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        
        self.fc_out = nn.Linear(in_channel*3, num_classes)

    def forward(self, x1,x2,x3,mod_code):     ####mod_code: b*4
        b,c,h,w,l = x1.shape # bxcx1x1x1
        X_ori = torch.cat((x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1)),1)  ###(b*3*c*h*w*l)
        X = torch.mean(X_ori.view(b,3,c,h*w*l),-1) ##b*4*c
        X1 = X.unsqueeze(1).repeat(1,3,1,1)  ##b*3*3*c
        X2 = X.unsqueeze(2).repeat(1,1,3,1)
        X_R = torch.cat((X1, X2),-1)   ###b*3*3*2c

        mod_code = mod_code.unsqueeze(-1).repeat(1, 1, 2 * c)
        X_R_1, X_R_2, X_R_3 = self.Relation1(torch.cat((X_R[:,0,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,1,:,:],mod_code),dim=-1)),\
                                     self.Relation1(torch.cat((X_R[:,2,:,:],mod_code),dim=-1))


        X_R_1,X_R_2,X_R_3 = F.softmax(X_R_1, 1),F.softmax(X_R_2, 1),F.softmax(X_R_3, 1)
        X_1_out = torch.matmul(X_ori.view(b,3,c,h*w*l).permute(0,2,3,1),X_R_1.permute(0,2,1).unsqueeze(-1)).squeeze(-1)  ##b*c*(h*w*l)*4 and b*c*4*1 -> b*c*(h*w*l)*1
        X_2_out = torch.matmul(X_ori.view(b, 3, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_2.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_3_out = torch.matmul(X_ori.view(b, 3, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_3.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)


        X_1_out,X_2_out,X_3_out = X_1_out.reshape(b,c,h,w,l),X_2_out.reshape(b,c,h,w,l),X_3_out.reshape(b,c,h,w,l)

        all_feat = torch.cat([x1+X_1_out,x2+X_2_out,x3+X_3_out], dim=1)
        all_feat = all_feat.squeeze()
        
        feat_out = self.fc_out(all_feat)
        
        return feat_out, all_feat
    
class pose_diff_code_block(nn.Module):
    def __init__(self, in_channel, out_feats=256, num_classes=5):
        super(pose_diff_code_block, self).__init__()
        self.Relation1 = nn.Sequential(
            nn.Linear(in_channel * 4, in_channel*2),
            nn.LeakyReLU(),
            nn.Linear(in_channel*2, in_channel)
            # nn.LeakyReLU(),
        )
        self.fc_out = nn.Linear(in_channel*4, out_feats)
        self.pose_out = nn.Linear(in_channel*4, num_classes)
        
        
    def forward(self, x1,x2,x3,x4,mod_code):     ####mod_code: b*4
        b,c,h,w,l = x1.shape # bxcx1x1x1
        X_ori = torch.cat((x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),x4.unsqueeze(1)),1)  ###(b*4*c*h*w*l)
        X = torch.mean(X_ori.view(b,4,c,h*w*l),-1) ##b*4*c
        X1 = X.unsqueeze(1).repeat(1,4,1,1)  ##b*4*4*c
        X2 = X.unsqueeze(2).repeat(1,1,4,1)
        X_R = torch.cat((X1, X2),-1)   ###b*4*4*2c

        mod_code = mod_code.unsqueeze(-1).repeat(1, 1, 2 * c)
        X_R_1, X_R_2, X_R_3, X_R_4 = self.Relation1(torch.cat((X_R[:,0,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,1,:,:],mod_code),dim=-1)),\
                                     self.Relation1(torch.cat((X_R[:,2,:,:],mod_code),dim=-1)),self.Relation1(torch.cat((X_R[:,3,:,:],mod_code),dim=-1)),


        X_R_1,X_R_2,X_R_3,X_R_4 = F.softmax(X_R_1, 1),F.softmax(X_R_2, 1),F.softmax(X_R_3, 1),F.softmax(X_R_4, 1)
        X_1_out = torch.matmul(X_ori.view(b,4,c,h*w*l).permute(0,2,3,1),X_R_1.permute(0,2,1).unsqueeze(-1)).squeeze(-1)  ##b*c*(h*w*l)*4 and b*c*4*1 -> b*c*(h*w*l)*1
        X_2_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_2.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_3_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_3.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)
        X_4_out = torch.matmul(X_ori.view(b, 4, c, h * w * l).permute(0, 2, 3, 1),
                               X_R_4.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1)

        X_1_out,X_2_out,X_3_out,X_4_out = X_1_out.reshape(b,c,h,w,l),X_2_out.reshape(b,c,h,w,l),X_3_out.reshape(b,c,h,w,l),X_4_out.reshape(b,c,h,w,l)

        all_feat = torch.cat([x1+X_1_out,x2+X_2_out,x3+X_3_out,x4+X_4_out], dim=1)
        all_feat = all_feat.squeeze()
        
        feat_out = self.fc_out(all_feat)
        pose_out = self.pose_out(all_feat)
        
        return pose_out, feat_out

        
class MFI(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(MFI, self).__init__()

        self.mod_diff_code_block = mod_diff_code_block(in_channel, num_classes=num_classes)
        self.pose_diff_code_block = pose_diff_code_block(in_channel, num_classes=num_classes)
        
    def forward(self, body, face, r_hand, l_hand, ecg, flow):
        
        B = body.shape[0]
        
        body = body.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        face = face.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        r_hand = r_hand.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        l_hand = l_hand.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        ecg = ecg.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        flow = flow.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        
        pose_out, pose_diff_out = self.pose_diff_code_block(body, face, r_hand,
                                                            l_hand, torch.ones((B,4)).to(DEVICE))
        if pose_out.shape[0] != B:
            pose_out = pose_out.unsqueeze(0)
            pose_diff_out = pose_diff_out.unsqueeze(0)
        # print(pose_out.shape, pose_diff_out.shape, len(pose_out.shape)) # torch.Size([5, 5]) torch.Size([5, 256])
        fuse_out, all_feat = self.mod_diff_code_block(flow, ecg,
                                            pose_diff_out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1),
                                            torch.ones((B,3)).to(DEVICE)) #.to(DEVICE)
        if fuse_out.shape[0] != B:
            fuse_out = fuse_out.unsqueeze(0)
        # print(pose_out.shape)
        self.pose_feats = pose_diff_out # B*256
        self.fusion_feats = all_feat # B*768 i.e., 256*3
        
        return pose_out, fuse_out

#%%
# mask_code = torch.from_numpy(np.random.randint(2, size=(2,3)))

# mfi = MFI(256, 5)

# body = torch.randn((5, 256)) #  node-0
# face = torch.randn((5, 256)) # node-1 
# r_hand = torch.randn((5, 256))# node-2
# l_hand = torch.randn((5, 256)) # node-3




# ecg = torch.randn((5, 256))
# flow = torch.randn((5, 256))


# y = mfi(body, face, r_hand, l_hand, ecg, flow)