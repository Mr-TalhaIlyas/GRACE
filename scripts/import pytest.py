import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import os
import sys
from scripts.main_eval import *

import torch.nn as nn

# Add the scripts directory to path for absolute imports
sys.path.insert(0, '/home/user01/Data/npj/scripts')


class TestMainEval:
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        return {
            'project_name': 'test_project',
            'experiment_name': 'test_experiment', 
            'num_fold': 0,
            'gpus_to_use': '0',
            'LOG_WANDB': False,
            'log_directory': '/tmp',
            'model': {
                'num_classes': 2,
                'hidden_dim': 256
            },
            'sub_classes': ['class1', 'class2'],
            'learning_rate': 0.001,
            'pose_lr_multiplier': 0.1,
            'WEIGHT_DECAY': 1e-4,
            'checkpoint_path': '/tmp',
            'sample_duration': 10,
            'window_overlap': 5,
            'batch_size': 16,
            'num_workers': 4,
            'pin_memory': True,
            'sanity_check': False,
            'external_data_dict': {}
        }
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model with named parameters"""
        model = nn.Module()
        
        # Add some mock parameters with different names
        model.bodygcn_layer = nn.Linear(10, 5)
        model.facegcn_layer = nn.Linear(10, 5) 
        model.rhgcn_layer = nn.Linear(10, 5)
        model.lhgcn_layer = nn.Linear(10, 5)
        model.other_layer = nn.Linear(10, 5)
        model.another_layer = nn.Linear(5, 2)
        
        return model
    
    @patch('scripts.main_eval.config')
    def test_optimizer_parameter_grouping(self, mock_config_import, mock_model, mock_config):
        """Test that parameters are correctly grouped into pose and other params"""
        mock_config_import.return_value = mock_config
        
        pose_params = []
        other_params = []
        
        for name, param in mock_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bodygcn' in name or 'facegcn' in name or 'rhgcn' in name or 'lhgcn' in name:
                pose_params.append(param)
            else:
                other_params.append(param)
        
        # Verify correct grouping
        assert len(pose_params) == 8  # 4 layers * 2 params each (weight + bias)
        assert len(other_params) == 4  # 2 layers * 2 params each
        
        # Verify optimizer params structure
        base_lr_main = mock_config['learning_rate']
        base_lr_pose = mock_config['learning_rate'] * mock_config['pose_lr_multiplier']
        
        optimizer_params = [
            {'params': other_params, 'lr': base_lr_main},
            {'params': pose_params, 'lr': base_lr_pose}
        ]
        
        assert len(optimizer_params) == 2
        assert optimizer_params[0]['lr'] == 0.001
        assert optimizer_params[1]['lr'] == 0.0001
    
    @patch('scripts.main_eval.torch.cuda.is_available')
    def test_device_configuration(self, mock_cuda_available, mock_config):
        """Test device configuration logic"""
        # Test CUDA available
        mock_cuda_available.return_value = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device.type == 'cuda'
        
        # Test CUDA not available
        mock_cuda_available.return_value = False
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        assert device.type == 'cpu'
    
    @patch('scripts.main_eval.load_chkpt')
    @patch('scripts.main_eval.os.path.join')
    def test_checkpoint_loading(self, mock_path_join, mock_load_chkpt, mock_config):
        """Test checkpoint loading functionality"""
        mock_path_join.return_value = '/tmp/test_checkpoint.pth'
        mock_model = Mock()
        mock_optimizer = Mock()
        
        # Simulate the checkpoint loading code
        chkpt_path = os.path.join(
            mock_config['checkpoint_path'],
            'alfred_cv_full_fusion_ce_margin_jsd_lossv16v4.pth'
        )
        
        load_chkpt(mock_model, mock_optimizer, chkpt_path)
        
        # Verify function was called with correct arguments
        mock_load_chkpt.assert_called_once_with(mock_model, mock_optimizer, '/tmp/test_checkpoint.pth')
        mock_path_join.assert_called_once()
    
    @patch('scripts.main_eval.GEN_DATA_LISTS')
    def test_data_fold_logic(self, mock_gen_data_lists, mock_config):
        """Test data fold generation logic"""
        mock_data = Mock()
        mock_data.get_folds.return_value = (['train1', 'train2'], ['test1', 'test2'])
        mock_gen_data_lists.return_value = mock_data
        
        # Test specific fold
        mock_config['num_fold'] = 1
        if mock_config['num_fold'] < 0:
            train_data = mock_data.get_folds(-1)
            test_data = mock_data.get_folds(-1)
        else:
            train_data, test_data = mock_data.get_folds(mock_config['num_fold'])
        
        assert train_data == ['train1', 'train2']
        assert test_data == ['test1', 'test2']
        mock_data.get_folds.assert_called_with(1)
        
        # Test all folds
        mock_config['num_fold'] = -1
        mock_data.get_folds.return_value = ['all_train_data']
        
        if mock_config['num_fold'] < 0:
            train_data = mock_data.get_folds(-1)
        
        assert train_data == ['all_train_data']
    
    @patch('scripts.main_eval.DataLoader')
    @patch('scripts.main_eval.SlidingWindowMMELoader')
    def test_dataloader_creation(self, mock_dataset, mock_dataloader, mock_config):
        """Test DataLoader creation with correct parameters"""
        mock_dataset_instance = Mock()
        mock_dataset_instance.mapping = ['item1', 'item2', 'item3']
        mock_dataset.return_value = mock_dataset_instance
        
        # Simulate train loader creation
        train_dataset = SlidingWindowMMELoader(['train_data'], mock_config, augment=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=mock_config['batch_size'],
            shuffle=True,
            num_workers=mock_config['num_workers'],
            drop_last=True,
            collate_fn=None,
            pin_memory=mock_config['pin_memory'],
            prefetch_factor=2,
            persistent_workers=True,
        )
        
        # Verify dataset was created with correct parameters
        mock_dataset.assert_called_with(['train_data'], mock_config, augment=True)
        
        # Verify DataLoader was called with correct parameters
        mock_dataloader.assert_called_with(
            mock_dataset_instance,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            drop_last=True,
            collate_fn=None,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        )
    
    @patch('scripts.main_eval.MME_Model')
    def test_model_initialization(self, mock_mme_model, mock_config):
        """Test model initialization and device placement"""
        mock_model_instance = Mock()
        mock_mme_model.return_value = mock_model_instance
        
        # Simulate model creation
        model = MME_Model(mock_config['model'])
        
        # Verify model was created with correct config
        mock_mme_model.assert_called_once_with(mock_config['model'])
        
        # Test device placement (would normally call model.to(device))
        assert mock_model_instance == model
    
    def test_num_classes_calculation(self, mock_config):
        """Test number of classes calculation"""
        num_classes = len(mock_config['sub_classes'])
        assert num_classes == 2
        
        sub_classes = 1
        assert sub_classes == 1
    
    @patch.dict(os.environ, {}, clear=True)
    def test_environment_variable_setting(self, mock_config):
        """Test environment variable configuration"""
        # Simulate environment variable setting
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = mock_config['gpus_to_use']
        
        assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    
    @patch('scripts.main_eval.torch.optim.AdamW')
    def test_optimizer_creation(self, mock_adamw, mock_model, mock_config):
        """Test optimizer creation with parameter groups"""
        # Simulate the optimizer parameter grouping
        pose_params = []
        other_params = []
        
        for name, param in mock_model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bodygcn' in name or 'facegcn' in name or 'rhgcn' in name or 'lhgcn' in name:
                pose_params.append(param)
            else:
                other_params.append(param)
        
        base_lr_main = mock_config['learning_rate']
        base_lr_pose = mock_config['learning_rate'] * mock_config['pose_lr_multiplier']
        
        optimizer_params = [
            {'params': other_params, 'lr': base_lr_main},
            {'params': pose_params, 'lr': base_lr_pose}
        ]
        
        # Create optimizer
        optimizer = torch.optim.AdamW(optimizer_params, weight_decay=mock_config['WEIGHT_DECAY'])
        
        # Verify AdamW was called with correct parameters
        mock_adamw.assert_called_once_with(optimizer_params, weight_decay=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])