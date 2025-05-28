# from configs import *


config = dict(
                gpus_to_use = '0',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = True,
                # alpha = 0.999,
                sanity_check = False,
                project_name= 'MME2',
                experiment_name= 'res1dcnn_f1',

                log_directory= "/home/user01/Data/mme/logs/",
                checkpoint_path= "/home/user01/Data/mme/chkpts/",

                pretrained_chkpts= None,#"/home/talha/Data/mme/scripts/models/pretrained/",

                folds =  '/home/user01/Data/mme/dataset/folds/',
                ecg_dir = '/home/user01/Data/mme/dataset/ecg_arr/',
                flow_dir = '/home/user01/Data/mme/dataset/flow/',
                pose_dir = '/home/user01/Data/mme/dataset/pose/',
                lbl_dir = '/home/user01/Data/mme/dataset/labels/',

                pin_memory=  True,
                num_workers= 12,# 2,,6

                num_fold = 1,

                # training settings
                batch_size= 8,

                # learning rate
                learning_rate= 0.001,
                lr_schedule= 'cos',
                num_repeats_per_epoch = 30,
                epochs= 100,
                warmup_epochs= 3,
                WEIGHT_DECAY= 0.0005,
                # AUX_LOSS_Weights= 0.4,

                # '''
                # Dataset
                # '''
                video_fps = 30, # FPS
                ecg_freq = 250, # Hz
                sample_duration = 10, # from [3,5,7,10]

                video_height= 224,
                video_width= 224,
                # sampling distribution
                alpha = 3,
                beta = 1,

                ignore_postictal = True,

                sub_classes = ['baseline', 'focal', 'tonic', 'clonic', 'pnes'],
                super_classes = ['baseline', 'gtcs', 'pnes'],
                # super_classes = ['baseline', 'seizure'],
                # ECG CWT settings
                steps = 128,
                wavelet = "mexh",


                # Model
                # model = config_model.mme
                # AUGMENTATIONS

                )