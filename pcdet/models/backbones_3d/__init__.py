from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import MANRNet, VirConvL8x
__all__ = {
    'MANRNet': MANRNet,
    'VirConvL8x': VirConvL8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
}
