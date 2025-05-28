#%%
import os
from configs.config import config

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_VISIBLE_DEVICES"] = config['gpus_to_use'];
from data.dataloader import GEN_DATA_LISTS
from configs.config import config
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from data.infer_loader import get_infer_sample
from torchmetrics import Accuracy
from data.utils import video_transform
from models.model import MME_Model
from tqdm import tqdm
from data.utils import values_fromreport
from sklearn.metrics import confusion_matrix, classification_report
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_accuracy(preds, lbls):
    
    lbls = np.clip(lbls, a_min=0, a_max=1)
    preds = np.clip(preds, a_min=0, a_max=1)
    
    matrix = confusion_matrix(lbls, preds,
                            labels=[0,1], normalize='true')
    ga, pa = matrix.diagonal()/matrix.sum(axis=1)

    return ga, pa, (ga+pa)/2

def get_precision_recall(preds, lbls):
    
    lbls = np.clip(lbls, a_min=0, a_max=1)
    preds = np.clip(preds, a_min=0, a_max=1)
    
    report = classification_report(lbls, preds,
                                    output_dict=True,
                                    zero_division=0)
    precision, recall, f1 = values_fromreport(report)
    return precision, recall, f1

class Inference(object):
    def __init__(self, dataset_dict, model):

        self.dataset_dict = dataset_dict
        self.model = model
        self.sup_accuracy = Accuracy(task="multiclass", num_classes=3)
    
    def calculate_accuracy(self, preds, labels):
        all_accuracy = {}
        for key in preds:
            aux_acc = self.sup_accuracy(preds[key].softmax(dim=-1).cpu().detach().squeeze(-1), # squeeze lass dim as batch==1
                                        labels.argmax(1).cpu().detach())
            all_accuracy[key] = aux_acc.item()
        return all_accuracy
    
    def get_sample(self, sample_idx, overlap_sec=3):
        self.data_sample, self.sliding_windows = get_infer_sample(self.dataset_dict,
                                                                    sample_idx,
                                                                    overlap_sec)
        return self.data_sample, self.sliding_windows

    def get_preds(self, batch_size=1, return_feats=False):
        
        #from numpy to torch and device
        for key in self.data_sample:
            if key != 'filename':
                self.data_sample[key] = torch.tensor(self.data_sample[key], dtype=torch.float32, device=DEVICE)
        
        fusion_preds, ecg_preds, pose_preds, flow_preds = [],[],[],[]
        all_lbls = []
        if return_feats:
            all_ecg_feats, all_of_feats, all_pose_feats, all_fuse_feats = [],[],[],[]
        # print('Running inference...')
        # predict on batches
        # for i in tqdm(range(0, len(self.sliding_windows), batch_size), desc='Infer'):
        for i in range(0, len(self.sliding_windows), batch_size):
            
            flow = video_transform(self.data_sample['frames'][i:i+batch_size]).to(DEVICE)
            # shape from N*C*T*V*M -> N*M*T*V*C
            body = self.data_sample['body'][i:i+batch_size].permute(0,4,2,3,1)
            face = self.data_sample['face'][i:i+batch_size].permute(0,4,2,3,1)
            rhand = self.data_sample['rh'][i:i+batch_size].permute(0,4,2,3,1)
            lhand = self.data_sample['lh'][i:i+batch_size].permute(0,4,2,3,1)
            # shape from B,S,T -> B,C,T,S
            # ecg = self.data_sample['ecg'][i:i+batch_size].unsqueeze(1).permute(0,1,3,2)#.to(DEVICE)
            # ViT ::: shape from B,T -> B,1,T
            ecg = self.data_sample['ecg'][i:i+batch_size].unsqueeze(1).type(torch.float).to(DEVICE)
            # sub_lbl_batch = self.data_sample['sub_lbls'][i:i+batch_size] # shape Bx5 NOT using
            sup_lbl_batch = torch.argmax(self.data_sample['sup_lbls'][i:i+batch_size], dim=1) # Bx3 -> Bx1
            
            self.model.eval()
            with torch.no_grad():
                preds = self.model.forward(flow, body, face, rhand, lhand, ecg)
                if return_feats:
                    all_ecg_feats.append(self.model.ecg_feats.detach().cpu().numpy().squeeze())
                    all_of_feats.append(self.model.of_feats.detach().cpu().numpy().squeeze())
                    if config['model']['fusion_type'] == 'graph':
                        all_pose_feats.append(self.model.fusion.pose_fusion.pose_feats.detach().cpu().numpy().squeeze())
                        all_fuse_feats.append(self.model.fusion.mod_fusion.fusion_feats.detach().cpu().numpy().squeeze())
                    elif config['model']['fusion_type'] == 'mfi':
                        all_pose_feats.append(self.model.fusion.mfi.pose_feats.detach().cpu().numpy().squeeze())
                        all_fuse_feats.append(self.model.fusion.mfi.fusion_feats.detach().cpu().numpy().squeeze())
                    elif config['model']['fusion_type'] == 'cmft':
                        all_pose_feats.append(self.model.fusion.cmft.pose_feats.detach().cpu().numpy().squeeze())
                        all_fuse_feats.append(self.model.fusion.cmft.fusion_feats.detach().cpu().numpy().squeeze())
                    elif config['model']['fusion_type'] == 'cat':
                        all_pose_feats.append(self.model.fusion.cat_fusion.pose_feats.detach().cpu().numpy().squeeze())
                        all_fuse_feats.append(self.model.fusion.cat_fusion.fusion_feats.detach().cpu().numpy().squeeze())
            # all_acc = self.calculate_accuracy(preds, sup_lbl_batch)

            fusion_preds.append(preds['fusion_outputs'].cpu().detach().numpy()) # Bx3
            ecg_preds.append(preds['ecg_outputs'].cpu().detach().numpy()) 
            pose_preds.append(preds['joint_pose_outputs'].cpu().detach().numpy())
            flow_preds.append(preds['flow_outpus'].cpu().detach().numpy())
            
            all_lbls.append(sup_lbl_batch.cpu().detach().numpy())
        
        fusion_preds = np.concatenate(fusion_preds, axis=0)
        ecg_preds = np.concatenate(ecg_preds, axis=0)
        pose_preds = np.concatenate(pose_preds, axis=0)
        flow_preds = np.concatenate(flow_preds, axis=0)
        
        all_lbls = np.concatenate(all_lbls, axis=0)
        
        if return_feats:
            all_ecg_feats = np.asarray(all_ecg_feats)
            all_of_feats = np.asarray(all_of_feats)
            all_pose_feats = np.asarray(all_pose_feats)
            all_fuse_feats = np.asarray(all_fuse_feats)
            
            return (fusion_preds.argmax(1), ecg_preds.argmax(1), pose_preds.argmax(1),
                    flow_preds.argmax(1), all_lbls, self.data_sample['filename'],
                    all_ecg_feats, all_of_feats, all_pose_feats, all_fuse_feats)
        else:
            return (fusion_preds.argmax(1), ecg_preds.argmax(1), pose_preds.argmax(1),
                    flow_preds.argmax(1), all_lbls, self.data_sample['filename'])
#%%
# model = MME_Model(config['model'])
# model.to(DEVICE)
# checkpoint = torch.load("/home/talha/Data/mme/chkpts/mme2.pth")#, map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
#%%

# data = GEN_DATA_LISTS(config)

# train_data, test_data = data.get_folds(config['num_fold'])

def run_inference(infer_data, infer_model, overlap_sec=3, return_feats=False):
    infer_model.eval()
    infer = Inference(infer_data, infer_model)

    files = []
    all_fusion_preds, all_ecg_preds, all_pose_preds, all_flow_preds, all_lbls = [],[],[],[],[]
    if return_feats:
        all_ecg_feats, all_of_feats, all_pose_feats, all_fuse_feats = [],[],[],[]
    for idx in range(len(infer_data['flow_paths'])):
        if (idx+1) % 4 == 0:
            print(f"Test Samples: {idx+1}/{len(infer_data['flow_paths'])}")
        data_sample, sliding_window = infer.get_sample(sample_idx=idx, overlap_sec=overlap_sec)
        if return_feats:
            fusion_preds, ecg_preds, pose_preds, flow_preds, lbls, file, ecg_feats, of_feats, pose_feats, fuse_feats = infer.get_preds(batch_size=1, return_feats=return_feats)
        else:
            fusion_preds, ecg_preds, pose_preds, flow_preds, lbls, file = infer.get_preds(batch_size=1, return_feats=return_feats)

        all_fusion_preds.append(fusion_preds)
        all_ecg_preds.append(ecg_preds)
        all_pose_preds.append(pose_preds)
        all_flow_preds.append(flow_preds)
        all_lbls.append(lbls)
        files.append(file)
        if return_feats:
            all_ecg_feats.append(ecg_feats)
            all_of_feats.append(of_feats)
            all_pose_feats.append(pose_feats)
            all_fuse_feats.append(fuse_feats)

    all_fusion_preds = np.concatenate(all_fusion_preds, axis=0)
    all_ecg_preds = np.concatenate(all_ecg_preds, axis=0)
    all_pose_preds = np.concatenate(all_pose_preds, axis=0)
    all_flow_preds = np.concatenate(all_flow_preds, axis=0)
    all_lbls = np.concatenate(all_lbls, axis=0)
    # files = np.concatenate(files, axis=0)

    if return_feats:
        all_ecg_feats = np.concatenate(all_ecg_feats, axis=0)
        all_of_feats = np.concatenate(all_of_feats, axis=0)
        all_pose_feats = np.concatenate(all_pose_feats, axis=0)
        all_fuse_feats = np.concatenate(all_fuse_feats, axis=0)


    # loop over all the preds and lbls to get accuracy and precision recall
    fba, fga, fpa = get_accuracy(all_fusion_preds, all_lbls)
    ecgba, ecgga, ecgpa = get_accuracy(all_ecg_preds, all_lbls)
    poseba, posega, posepa = get_accuracy(all_pose_preds, all_lbls)
    flowba, flowga, flowpa = get_accuracy(all_flow_preds, all_lbls)

    fprecision, frecall, ff1 = get_precision_recall(all_fusion_preds, all_lbls)
    ecgprecision, ecgrecall, ecgf1 = get_precision_recall(all_ecg_preds, all_lbls)
    poseprecision, poserecall, posef1 = get_precision_recall(all_pose_preds, all_lbls)
    flowprecision, flowrecall, flowf1 = get_precision_recall(all_flow_preds, all_lbls)

    if return_feats:
        # then return the feats as well along with all labels for tSNE plotting
        return {'filenames': files,
                'fusion': [fba, fga, fpa, fprecision, frecall, ff1],
                'ecg': [ecgba, ecgga, ecgpa, ecgprecision, ecgrecall, ecgf1],
                'pose': [poseba, posega, posepa, poseprecision, poserecall, posef1],
                'flow': [flowba, flowga, flowpa, flowprecision, flowrecall, flowf1],
                'all_lbls': all_lbls,
                'all_fusion_preds': all_fusion_preds,
                'all_ecg_preds': all_ecg_preds,
                'all_pose_preds': all_pose_preds,
                'all_flow_preds': all_flow_preds,
                'all_ecg_feats': all_ecg_feats,
                'all_flow_feats': all_of_feats,
                'all_pose_feats': all_pose_feats,
                'all_fusion_feats': all_fuse_feats
                }
    else:
        return {'fusion': [fba, fga, fpa, fprecision, frecall, ff1],
                'ecg': [ecgba, ecgga, ecgpa, ecgprecision, ecgrecall, ecgf1],
                'pose': [poseba, posega, posepa, poseprecision, poserecall, posef1],
                'flow': [flowba, flowga, flowpa, flowprecision, flowrecall, flowf1]
                }










#%%

# labels = all_sup_lbls
# predictions = all_sup_preds#np.argmax(all_sup_preds, -1)


# thresholded_predictions = np.where(predictions > 0.7, 1, 0)

# import matplotlib.pyplot as plt

# # Set up the plot
# plt.figure(figsize=(5, 3))

# # Plot raw predictions
# # plt.plot(predictions, label='Raw Predictions', color='skyblue', linestyle='--')

# # Highlight the ambiguous region between 0.5 and 0.7
# ambiguous_indices = np.where((predictions > 0.5) & (predictions <= 0.7))
# # plt.plot(ambiguous_indices[0], predictions[ambiguous_indices], 'o', label='Ambiguous Predictions (0.5-0.7)', color='orange', markersize=8)

# # Plot thresholded predictions (considering them as step for clearer visualization)
# times = np.arange(len(predictions))
# plt.step(times, thresholded_predictions, label='Thresholded Predictions', color='red', where='mid')

# # Plot original labels
# # plt.step(times, labels + 0.05, label='Original Labels', color='green', where='mid', linestyle=':')
# plt.step(times, labels, label='Original Labels', color='green')
# # Making plot look nice
# plt.yticks([0, 1], ['0', '1'])
# plt.fill_between(times, 0, 1, where=(predictions > 0.5) & (predictions <= 0.7), color='orange', alpha=0.1, step='mid')
# plt.legend(loc='upper left')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')
# plt.title('Network Predictions vs. Original Labels')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# # Show plot
# plt.tight_layout()
# plt.show()

# #%%
# x = np.argmax(all_sub_preds, axis=1)
