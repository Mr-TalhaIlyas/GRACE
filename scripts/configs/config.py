from configs import *
from configs import config_model


config = dict(
                gpus_to_use = '0',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = False,
                ema_momentum = 0.999,
                sanity_check = False,
                project_name= 'NPJ',
                experiment_name= 'alfred_cv_full_fusion_ce_margin_jsd_lossv18ema',

                log_directory= "/home/user01/Data/npj/logs/",
                checkpoint_path= "/home/user01/Data/npj/chkpts/",

                pretrained_chkpts= "/home/user01/Data/npj/scripts/models/pretrained/",

                folds =  '/home/user01/Data/npj/datasets/alfred/folds/',
                ecg_dir = '/home/user01/Data/npj/datasets/alfred/cv/ecg/',
                flow_dir = '/home/user01/Data/npj/datasets/alfred/cv/flow/',
                pose_dir = '/home/user01/Data/npj/datasets/alfred/cv/pose/',
                lbl_dir = '/home/user01/Data/npj/datasets/alfred/cv/labels/',
                
                # create external data dirs in external_data_dict
                external_data_dict = {
                    'ecg_dir': '/home/user01/Data/npj/datasets/alfred/external/ecg/',
                    'flow_dir': '/home/user01/Data/npj/datasets/alfred/external/flow/',
                    'pose_dir': '/home/user01/Data/npj/datasets/alfred/external/pose/',
                    'lbl_dir': '/home/user01/Data/npj/datasets/alfred/external/labels/',
                },

                # TUH DATASET
                tuh_data_dir = '/home/user01/Data/npj/datasets/tuh/',
                # SeizeIT2 DATASET
                seizeit2_data_dir = '/home/user01/Data/npj/datasets/seizeit2/',
                # BioSignal Chkpts base directory
                bio_signal_chkpts_dir = '/home/user01/Data/npj/scripts/ts/chkpt/',
                pin_memory=  True,
                num_workers= 12,# 2,,6

                num_fold = -1, # -1 for all folds, 0 for fold 0, 1 for fold 1, etc.

                # training settings
                batch_size= 1,

                # learning rate
                learning_rate= 0.000001, # 0.001
                pose_lr_multiplier= 0.5,#0.1,
                lr_schedule= 'cos', # cos cyclic
                max_lr = 0.0001, # e.g., max_lr will be 6x base_lr for each group
                base_lr = 0.000001, # e.g., base_lr will be 1x base_lr for each group
                clr_step_epochs= 4,       # Number of epochs for one half-cycle (base_lr to max_lr)
                clr_mode='triangular',    # 'triangular', 'triangular2', or 'exp_range'
                
                epochs= 100,
                warmup_epochs= 1,
                WEIGHT_DECAY= 0.01,
                
                # AUX_LOSS_Weights= 0.4,

                # '''
                # Dataset
                # '''
                video_fps = 30, # FPS
                ecg_freq = 250, # Hz
                sample_duration = 10, # in seconds from [3,5,7,10]
                window_overlap = 5, # in seconds

                flow_frames = 48,
                pose_frames = 150,
                video_height= 224,
                video_width= 224,
                # sampling distribution
                alpha = 3,
                beta = 1,

                ignore_postictal = True,

                sub_classes = ['baseline', 'focal', 'tonic', 'clonic', 'pnes'],
                # super_classes = ['baseline', 'gtcs', 'pnes'],
                super_classes = ['baseline', 'seizure'],
                
                LABEL_MAP_TUH = {"fnsz": 0,
                                 "tcsz": 1,
                                 "nesz": 2,
                                 "bckg": -1
                                 },
                LABEL_MAP_SeizeIT2 = {"sz_foc_f2b": 0,
                                      "sz_gen_m_tonicClonic": 1,
                                      "bckg": -1
                                      },
                # ECG CWT settings
                steps = 128,
                wavelet = "mexh",
                
                # -LOSSES
                auxiliary_loss_weight = 0.2, # for ecg and flow
                main_loss_weight = 1.0,
                # due to worse perfromance and NOISy pose mdoalitye reduce the weight
                joint_pose_weight = 0.05,
                pose_fhb_weight = 0.05,
                
                consistent_loss_weight = 0.2,
                inter_group_consistency_weight = 0.4,
                
                fusion_warmup_epochs = 2,
                neg_jsd_factor = 0.5,
                neg_margin_weight=0.3,
                neg_margin=0.2,
                
                neg_margin_factor_ecg_vs_flow= 1.0,
                neg_margin_factor_flow_vs_ecg= 1.0,
                neg_margin_factor_ecg_vs_pose= 0.5,
                neg_margin_factor_pose_vs_ecg= 0.5,
                # neg_margin_factor_flow_vs_pose= 0.5,
                # neg_margin_factor_pose_vs_flow= 1.0,
                # Example: Add to your config
                recall_thresholds = {
                    'flow': 0.80,  # Target recall for flow modality
                    'ecg': 0.75,   # Target recall for ECG modality
                    'pose': 0.75   # Target recall for pose (will freeze all GCNs)
                },
                encoder_module_names = {
                    'flow': ['slowfast'],  # model.slowfast
                    'ecg': ['ewt'],       # model.ewt
                    'pose': ['bodygcn', 'facegcn', 'rhgcn', 'lhgcn'] # model.bodygcn, etc.
                },
                # Model
                model = config_model.mme,
                # AUGMENTATIONS
                log_step_freq = 1,
                val_every = 1,
                )