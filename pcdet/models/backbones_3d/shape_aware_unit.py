import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional

try:
    import spconv.pytorch as spconv
    SPCONV_AVAILABLE = True
except ImportError:
    SPCONV_AVAILABLE = False

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
    
    spconv = type('MANRSpconv', (), {'SparseConvTensor': MANRSparseConvTensor})()

class ShapeAwareUnit(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(ShapeAwareUnit, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.shape_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )

        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim // 2 + input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.residual_proj = None
        if input_dim != output_dim:
            self.residual_proj = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: Tensor) -> Tensor:

        batch_size, num_points, input_dim = x.shape

        shape_features = self.shape_encoder(x)  # [batch_size, num_points, hidden_dim//2]

        enhanced_features = self.feature_enhancer(
            torch.cat([x, shape_features], dim=-1)
        )

        if self.residual_proj is not None:
            residual = self.residual_proj(x)
        else:
            residual = x
            
        output = enhanced_features + residual
        
        return output
    
    def get_output_dim(self) -> int:

        return self.output_dim
    
    def get_input_dim(self) -> int:

        return self.input_dim


class ShapeAwareUnit3D(nn.Module):

    def __init__(self, input_channels: int, output_channels: int, hidden_channels: int = 64):

        super(ShapeAwareUnit3D, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels

        self.shape_aware_conv = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.feature_fusion = nn.Sequential(
            nn.Conv3d(hidden_channels + input_channels, output_channels, 1),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True)
        )

        if input_channels != output_channels:
            self.residual_conv = nn.Conv3d(input_channels, output_channels, 1)
        else:
            self.residual_conv = None
    
    def forward(self, x: Tensor) -> Tensor:

        shape_features = self.shape_aware_conv(x)  # [batch_size, hidden_channels, D, H, W]

        fused_features = self.feature_fusion(
            torch.cat([x, shape_features], dim=1)
        )
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
        else:
            residual = x
            
        output = fused_features + residual
        
        return output


if __name__ == "__main__":

    shape_unit = ShapeAwareUnit(input_dim=128, hidden_dim=64, output_dim=128)
    test_input = torch.randn(2, 1024, 128)
    test_output = shape_unit(test_input)

    shape_unit_3d = ShapeAwareUnit3D(input_channels=16, output_channels=16)
    test_input_3d = torch.randn(2, 16, 32, 32, 32)
    test_output_3d = shape_unit_3d(test_input_3d)