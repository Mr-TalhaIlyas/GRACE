

pretrained_root_dir = '/home/user01/Data/npj/scripts/models/pretrained/'

mme = dict(
            batch_size= 32,

            num_sub_classes = 5,
            num_sup_classes = 2, # wiht sigmoid activation output
            
            fusion_type = 'graph', # 'cat', 'trans', 'graph', 'cmft'
            slowfast_pretrained_chkpts= None,#f"{pretrained_root_dir}slowfast_r50_4x16x1_kinetics400-rgb.pth",

            num_persons=1,
            backbone_in_channels=3,
            head_in_channels=256,
            # gcn_body_17kpts_kinetic400
            # gcn_hand_21kpts_fphad45
            body_pretrainned_chkpts= None,#f"{pretrained_root_dir}ctrgcn_body_17kpts.pth",
            hand_pretrainned_chkpts= None,#f"{pretrained_root_dir}gcn_hand_21kpts_fphad45.pth",
            face_pretrainned_chkpts= None,

            ewt_head_ch = 768,
            mod_feats = 512,
            ewt_dropout_ratio = 0.3,
            # ecg_vit_fold1flip
            ewt_pretrainned_chkpts=None,#f'{pretrained_root_dir}best_tuh_ecgH_bvg_InceptionTime.pth',

            # INceptionTime
            hrv_channels=19,
            incep_num_features=64,
            
            xcm_num_features=128,
            # GTN dual branch
            d_input_T = 2500, # frequency * 10s
            d_model_emb = 256,
            d_hidden_ffn = 512,
            gtn_num_heads = 8,
            gtn_num_layers = 8,

            
            fusion_in_channels = 512, # inception time has best on 256
            fusion_heads = 16,
            # DROPOUTS
            flow_dropout=0.5,
            pose_dropout=0.5,
            ecg_dropout=0.5, # ecg or hrv
            
            pose_fusion_dropout=0.5, # body, face, rh, lh pose joint graph
            mod_fusion_dropout=0.5,
            
            # 
            )
