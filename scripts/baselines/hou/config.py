config = dict(
                gpus_to_use = '0',
                DPI = 300,
                LOG_WANDB= False,
                BENCHMARK= False,
                DEBUG = False,
                USE_EMA_UPDATES = False,
                ema_momentum = 0.999,
                sanity_check = False,
                project_name= 'MME2_baselines',
                experiment_name= 'hou_fold1',

                log_directory= "/home/user01/Data/mme/baseline_op/logs/",
                checkpoint_path= "/home/user01/Data/mme/baseline_op/chkpts/",

                pretrained_chkpts= "/home/user01/Data/mme/scripts/models/pretrained/",

                folds =  '/home/user01/Data/mme/dataset/folds/',
                flow_dir = '/home/user01/Data/mme/dataset/flow/',
                pose_dir = '/home/user01/Data/mme/dataset/baseline_feats/hou/pose/',
                lbl_dir = '/home/user01/Data/mme/dataset/labels/',

                pin_memory=  True,
                num_workers= 12,# 2,,6

                num_fold = 2,
                num_epochs = 100,
                # training settings
                batch_size= 8,

                # learning rate
                learning_rate= 10e-3,
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

                video_height= 112,
                video_width= 112,
                # sampling distribution
                alpha = 5.0,
                beta = 4.0,

                ignore_postictal = True,

                sub_classes = ['baseline', 'focal', 'tonic', 'clonic', 'pnes'],
                # super_classes = ['baseline', 'gtcs', 'pnes'],
                super_classes = ['baseline', 'seizure'],
                # ECG CWT settings
                steps = 128,
                wavelet = "mexh",

                )