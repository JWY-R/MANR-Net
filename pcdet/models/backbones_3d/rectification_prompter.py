import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

class MANRSparseConvTensor:
    def __init__(self, features, indices, spatial_shape, batch_size):
        self.features = features
        self.indices = indices
        self.spatial_shape = spatial_shape
        self.batch_size = batch_size
        
    def replace_feature(self, new_features):
        return MANRSparseConvTensor(
            new_features, self.indices, self.spatial_shape, self.batch_size
        )


class MANRSpconv:
    SparseConvTensor = MANRSparseConvTensor

spconv = MANRSpconv()

class RectificationPrompterLayer(nn.Module):
    def __init__(self, 
                 input_c: int = 16, 
                 output_c: int = 16, 
                 stride: int = 1, 
                 padding: int = 1, 
                 indice_key: str = 'vir1',
                 conv_depth: bool = False):

        super().__init__()
        self.stride = stride
        self.indice_key = indice_key
        self.conv_depth = conv_depth

        try:
            from Point_MAE_pretask_dev import (
                Group, PositionalEmbedding, PointNetSetAbstraction, PointNetFeaturePropagation
            )
        except ImportError:
            class MANRGroup:
                def __init__(self, num_group, group_size):
                    self.num_group = num_group
                    self.group_size = group_size
                    
            class MANRPositionalEmbedding:
                def __init__(self, embedding_level):
                    self.embedding_level = embedding_level
                    
            class MANRPointNetSetAbstraction:
                def __init__(self, num_group, group_size, hidden_dimension, mlp):
                    pass
                    
            class MANRPointNetFeaturePropagation:
                def __init__(self, in_channel, mlp):
                    pass
                    
            Group = MANRGroup
            PositionalEmbedding = MANRPositionalEmbedding
            PointNetSetAbstraction = MANRPointNetSetAbstraction
            PointNetFeaturePropagation = MANRPointNetFeaturePropagation
        

        self.num_group = 32
        self.group_size = 16
        self.top_center_dim = 12
        self.hidden_dimension = 384
        self.embedding_level = 4
        
        self.group_divider = Group(self.num_group, self.group_size)
        self.position_embedding = PositionalEmbedding(self.embedding_level)

        self.abstraction = PointNetSetAbstraction(
            self.num_group, self.group_size, self.hidden_dimension, 
            mlp=[64, 32, self.top_center_dim]
        )

        self.propagation1 = PointNetFeaturePropagation(
            in_channel=input_c*(2*self.embedding_level+1)+32, 
            mlp=[32, 32]
        )
        self.propagation2 = PointNetFeaturePropagation(
            in_channel=self.top_center_dim, 
            mlp=[64, 32]
        )

        self.score_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_c),
        )
        self.score_factor = 1.0

        for layer in self.score_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=5.0**0.5)
                nn.init.constant_(layer.bias, val=0.0)
    
    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):

        features = sp_tensor.features
        indices = sp_tensor.indices
        spatial_shape = sp_tensor.spatial_shape
        return sp_tensor
    
    def get_output_dim(self) -> int:
        return self.out_channels
    
    def get_input_requirements(self) -> dict:
        return {
            'x': {'shape': '(B, N, C)', 'type': 'Tensor'},
            'center1': {'shape': '(B, M, 3)', 'type': 'Tensor'}, 
            'center1_feature': {'shape': '(B, M, C)', 'type': 'Tensor'}
        }

try:
    from pcdet.utils import common_utils
    from pcdet.models.backbones_3d import __init__ as backbones_init


    if hasattr(backbones_init, 'MODELS'):
        backbones_init.MODELS.register_module()(RectificationPrompterLayer)
except ImportError:
    pass