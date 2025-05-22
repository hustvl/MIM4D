from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet
from .. import utils
from torchvision.utils import save_image
import torch.nn.functional as F
import random
import cv2
import copy
from torch.nn.init import constant_
from mmdet3d.models import builder
import os

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

@DETECTORS.register_module()
class UVTRSSL(MVXTwoStageDetector):
    """UVTR."""

    def __init__(
        self,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        depth_head=None,
        view_trans=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        render_conv_cfg=None,
        pretrained=None,
        cam_sweep_num=None,
        HOP=None,
        history_decoder=None,
    ):
        super(UVTRSSL, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        if view_trans:
            vtrans_type = view_trans.pop("type", "Uni3DVoxelPoolDepth")
            self.view_trans = getattr(utils, vtrans_type)(
                **view_trans
            )  # max pooling, deformable detr, bilinear
        if self.with_img_backbone:
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if depth_head is not None:
                depth_dim = self.view_trans.depth_dim
                dhead_type = depth_head.pop("type", "SimpleDepth")
                if dhead_type == "SimpleDepth":
                    self.depth_net = Conv2d(out_channels, depth_dim, kernel_size=1)
                else:
                    self.depth_net = DepthNet(
                        out_channels, out_channels, depth_dim, **depth_head
                    )
            self.depth_head = depth_head

        if pts_middle_encoder:
            self.pts_fp16 = (
                True if hasattr(self.pts_middle_encoder, "fp16_enabled") else False
            )

        self.render_conv = nn.Sequential(
            nn.Conv3d(
                render_conv_cfg["in_channels"],
                render_conv_cfg["out_channels"],
                kernel_size=render_conv_cfg["kernel_size"],
                padding=render_conv_cfg["padding"],
                stride=1,
            ),
            nn.BatchNorm3d(render_conv_cfg["out_channels"]),
            nn.ReLU(inplace=True),
        )
        self.HOP = HOP
        if self.HOP:
            self.cam_sweep_num = cam_sweep_num
            self.history_decoder = builder.build_backbone(history_decoder)

        # self._reset_parameters()
        self.ssim = SSIM()
        self.idx = 0

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        constant_(self.sampling_offsets.bias.data, 0.)

    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, "depth_head") and self.depth_head is not None

    @force_fp32()
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        if not self.pts_voxel_encoder or pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        if not self.pts_fp16:
            voxel_features = voxel_features.float()

        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            # img_feat = img_feat.float()
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"))
    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        if hasattr(self, "img_backbone"):
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(
                img=img, img_metas=img_metas, img_feats=img_feats
            )
        else:
            img_feats, img_depth = None, None

        if hasattr(self, "pts_voxel_encoder"):
            pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        else:
            pts_feats = None

        return pts_feats, img_feats, img_depth

    @auto_fp16(apply_to=("img"))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None or not self.with_depth_head:
            return None
        B, N, C, H, W = img.size()
        img = img.view(B * N, C, H, W)
        depth = []
        for _feat in img_feats:
            _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))
            _depth = _depth.softmax(dim=1)
            depth.append(_depth)
        return depth

    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
        self, uni_feats, points, img, img_metas, img_depth, key_id, batch_rays
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        out_dict = self.pts_bbox_head(
            uni_feats, batch_rays, img_metas
        )
        losses = self.pts_bbox_head.loss(out_dict, batch_rays)
        return losses, None


    @force_fp32(apply_to=("img", "points"))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def view_transform(self, img_feats, img_depth, pts_feats, img_metas):
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        B,N,D,Z,H,W = uni_feats.shape
        uni_feats = uni_feats.view(B*N,D,Z,H,W)
        uni_feats = self.render_conv(uni_feats)
        uni_feats = uni_feats.view(B,N,*uni_feats.shape[1:])
        return uni_feats
    
    @force_fp32()
    def shift_lidarpoints(self, points, img_metas, target_id, curr_id):
        pts = points[0]

        pts_xyz = pts[:, :3]
        view_point = pts_xyz.view(-1,3).cpu().numpy()
        np.save("pts.npy",view_point)
        lidar2ego = img_metas[0]['lidar2ego'][0]
        ego2global = img_metas[0]['ego2global'][0]
        currlidar2currego = torch.tensor(lidar2ego[curr_id],dtype=pts.dtype,device=pts.device)
        targetlidar2targetego = torch.tensor(lidar2ego[target_id],dtype=pts.dtype,device=pts.device)
        currego2global = torch.tensor(ego2global[curr_id],dtype=pts.dtype,device=pts.device)
        targetego2global = torch.tensor(ego2global[target_id],dtype=pts.dtype,device=pts.device)

        global2currego = currego2global.inverse()
        currego2currlidar = currlidar2currego.inverse()
        currlidar2targetlidar = (currego2currlidar @ global2currego @ targetego2global @ targetlidar2targetego).inverse()
        pts_xyz = torch.cat([pts_xyz, torch.ones_like(pts_xyz[:, :1])], dim=-1)
        pts_xyz = currlidar2targetlidar.matmul(pts_xyz.unsqueeze(-1))
        view_point = pts_xyz[:, :3, 0].view(-1,3).cpu().numpy()
        np.save("pts_shifted.npy",view_point)
        pts[:, :3] = pts_xyz[:, :3, 0]

        points[0] = pts

        return points

    @force_fp32()
    def shift_feature(self, input, img_metas, adj_id, key_id):
        # n, c, h, w = input.shape

        n,c,z,h,w = input.shape
        lidar2ego = img_metas[0]['lidar2ego'][0]
        ego2global = img_metas[0]['ego2global'][0]
        keylidar2keyego = torch.tensor(lidar2ego[key_id],dtype=input.dtype,device=input.device)
        sweeplidar2sweepego = torch.tensor(lidar2ego[adj_id],dtype=input.dtype,device=input.device)
        keyego2global = torch.tensor(ego2global[key_id],dtype=input.dtype,device=input.device)
        sweepego2global = torch.tensor(ego2global[adj_id],dtype=input.dtype,device=input.device)

        global2keyego = keyego2global.inverse()
        keyego2keylidar = keylidar2keyego.inverse()
        keylidar2sweeplidar = (keyego2keylidar @ global2keyego @ sweepego2global @ sweeplidar2sweepego).inverse()

        # generate voxel
        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, 1, w).expand(z, h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(1, h, 1).expand(z, h, w)
        zs = torch.linspace(
            0, z - 1, z, dtype=input.dtype,
            device=input.device).view(z, 1, 1).expand(z, h, w)
        voxel = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1)
        voxel = voxel.view(1, z, h, w, 4).expand(n, z, h, w, 4).view(n, z, h, w, 4, 1)

        voxel = keylidar2sweeplidar.matmul(voxel)
        normalize_factor = torch.tensor([w - 1.0 ,h - 1.0, z - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        voxel = voxel[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        output = F.grid_sample(input, voxel.to(input.dtype), align_corners=True)
        return output
    
    def adjust_img_metas(self,img_ori, img_depths_ori, img_metas_ori, keyid):
        img = img_ori
        img_depths = img_depths_ori
        img_metas = copy.deepcopy(img_metas_ori)
        for img_meta in img_metas:
            #lidar2img
            lidar2imgs = img_meta['lidar2img']
            lidar2imgs = [lidar2img[keyid:keyid+1] for lidar2img in lidar2imgs]
            img_meta['lidar2img'] = lidar2imgs
            #keylidar2sweepimg
            keylidar2sweepimg = img_meta['keylidar2sweepimg']
            keylidar2sweepimg = [keylidar2img[keyid:keyid+1] for keylidar2img in keylidar2sweepimg]
            img_meta['keylidar2sweepimg'] = keylidar2sweepimg    

            #lidar2cam
            lidar2cams = img_meta['lidar2cam']
            lidar2cams = [lidar2cam[keyid:keyid+1] for lidar2cam in lidar2cams]
            img_meta['lidar2cam'] = lidar2cams
            #keylidar2sweepcam
            keylidar2sweepcam = img_meta['keylidar2sweepcam']
            keylidar2sweepcam = [keylidar2cam[keyid:keyid+1] for keylidar2cam in keylidar2sweepcam]
            img_meta['keylidar2sweepcam'] = keylidar2sweepcam   

            pad_before_shapes = img_meta['pad_before_shape'][keyid*6:keyid*6+6]
            img_meta['pad_before_shape'] = pad_before_shapes

            img_shape = img_meta['img_shape'][keyid*6:keyid*6+6]
            img_meta['img_shape'] = img_shape

        if img is not None:
            img = img.view(1,6,self.cam_sweep_num,*img.shape[2:])
            img = img[:, :,keyid]       

        if img_depths is not None:
            img_depths = [img_depth[keyid*6:keyid*6+6] for img_depth in img_depths]

        return img, img_depths, img_metas
    
    def shift_voxel_feature_hop(self, uni_feats, img_metas, key_id, num_sweep):
        voxel_feat_list = []
        for adj_id in range(num_sweep):
            if adj_id == key_id:
                continue
            voxel_feat = self.shift_feature(uni_feats[:,adj_id], img_metas, adj_id, key_id) #(n, d, z, h, w)
            n,c,z,h,w = voxel_feat.shape
            voxel_feat = voxel_feat.view(n,c*z,h,w)
            voxel_feat_list.append(voxel_feat)
        return voxel_feat_list
    
    def point_filter(self,pts):
        points = pts[0]

        pts_xyz = points[:, :3]
        # pc_range = self.pts_bbox_head.pc_range
        redius = torch.norm(pts_xyz, dim=-1, keepdim=False)
        mask = redius > 2.5
        points = points[mask]
        pts[0] = points
        return pts

    def get_batchrays(self, img, points, img_metas, is_single=False):

        if is_single==False:
            id_list = list(range(1, self.cam_sweep_num))
            key_id = random.choice(id_list)
            self.new_id_list = [0, key_id]
        else:
            self.new_id_list = [0]
        batch_rays_list = []
        for i in self.new_id_list:
            img_adj, img_depth_adj, img_metas_adj = self.adjust_img_metas(img, None, img_metas, keyid=i)
            batch_rays = self.pts_bbox_head.sample_rays(points, img_adj, img_metas_adj)
            batch_rays_list.append(batch_rays)

        return batch_rays_list
    
    def random_masking(self,imgs, batch_rays_list,img_metas):
        imgs = imgs.view(6*self.cam_sweep_num,*imgs.shape[2:])
        imgs = self.img_backbone.random_mask(imgs)
        imgs = imgs.view(6,self.cam_sweep_num,*imgs.shape[1:])
        for i in range(self.cam_sweep_num):
            if i in self.new_id_list:
                image = imgs[:, i]
                sampled_pts_cam = batch_rays_list[self.new_id_list.index(i)][0]["sampled_pts_cam"]
                for c_idx in range(image.shape[0]):
                    H, W = image.shape[2:]
                    stride = 8
                    mask = torch.ones(1, H//stride, W//stride, dtype=torch.bool, device=image.device)
                    mask[:,sampled_pts_cam[c_idx][:,1].long()//stride,sampled_pts_cam[c_idx][:,0].long()//stride] = False
                    mask = mask.repeat_interleave(stride, 1).repeat_interleave(stride, 2)
                    image[c_idx] = image[c_idx]*mask
                imgs[:, i] = image
            else:
                continue
        return imgs.view(1, 6*self.cam_sweep_num,*imgs.shape[2:])
    
    
    def random_masking_single(self,imgs, batch_rays_list,img_metas):
        imgs = imgs.view(6, 1,*imgs.shape[2:])
        imgs = imgs[:, 0]
        imgs = self.img_backbone.random_mask(imgs)
    
        return imgs.view(1, 6,*imgs.shape[1:])   
    

    def get_adj_loss_hop(self, uni_feats, img, img_depth, points, img_metas, losses, batch_rays_list):
        num_sweep = uni_feats.shape[1]
        loss_adj = {}
        loss = {}

        for i, key_id in enumerate(self.new_id_list):

            pre_voxel_feat_list = self.shift_voxel_feature_hop(uni_feats, img_metas, key_id, num_sweep)
            pre_voxel_feat = self.history_decoder(pre_voxel_feat_list)


            # img_depth is not used
            img_adj, img_depth_adj, img_metas_adj = self.adjust_img_metas(img, img_depth, img_metas, keyid=key_id)

            losses_pts_adj, shifted_depth_pred = self.forward_pts_train(pre_voxel_feat, points, img_adj, img_metas_adj, img_depth_adj, key_id, batch_rays_list[i])           
            # direct loss
            curr_voxel_feat = uni_feats[:,key_id]
            losses_pts, direct_depth_pred = self.forward_pts_train(curr_voxel_feat, points, img_adj, img_metas_adj, img_depth_adj, key_id, batch_rays_list[i])  

            
            if i == 0:
                losses_pts_adj['rgb_loss_adj'] = losses_pts_adj['rgb_loss']
                losses_pts_adj['depth_loss_adj'] = losses_pts_adj['depth_loss']
                losses_pts_adj.pop('rgb_loss')
                losses_pts_adj.pop('depth_loss')
                loss_adj.update(losses_pts_adj)
                loss.update(losses_pts)
            else:
                loss_adj['rgb_loss_adj'] += losses_pts_adj['rgb_loss']
                loss_adj['depth_loss_adj'] += losses_pts_adj['depth_loss']
                loss['rgb_loss'] += losses_pts['rgb_loss']
                loss['depth_loss'] += losses_pts['depth_loss']

        losses.update(loss_adj)
        losses.update(loss)
        for key in losses.keys():
            losses[key] /= len(self.new_id_list)
        return losses





    def forward_train(self, points=None, img_metas=None, img=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.HOP:
            is_hop = True
            lidar2ego = img_metas[0]['lidar2ego'][0]
            ego2global = img_metas[0]['ego2global'][0]
            for i in range(len(img_metas[0]['keylidar2sweepimg'])):
                keylidar2sweepimg = img_metas[0]['keylidar2sweepimg'][i]
                if keylidar2sweepimg.shape[0] != self.cam_sweep_num:
                    is_hop = False
                    break
            if lidar2ego.shape[0] == 1 or ego2global.shape[0] != self.cam_sweep_num :
                is_hop = False
            if is_hop:
                points = self.point_filter(points)
                batch_rays_list = self.get_batchrays(img, points, img_metas)
                # if use maskresnet , open this
                # img = self.random_masking(img, batch_rays_list,img_metas)
                pts_feats, img_feats, img_depth = self.extract_feat(
                    points=points, img=img, img_metas=img_metas
                )
                uni_feats = self.view_transform(img_feats, img_depth, pts_feats, img_metas) # b n c z h w
                losses = dict()
                losses = self.get_adj_loss_hop(uni_feats, img, img_depth, points, img_metas, losses, batch_rays_list)

                return losses
            else:            
                batch_rays_list = self.get_batchrays(img, points, img_metas, is_single=True)
                
                img, _, img_metas = self.adjust_img_metas(img_ori=img, img_depths_ori=None, img_metas_ori=img_metas, keyid=0)
                # img = self.random_masking_single(img, batch_rays_list,img_metas)
                pts_feats, img_feats, img_depth = self.extract_feat(
                points=points, img=img, img_metas=img_metas
                )
                uni_feats = self.view_transform(img_feats, img_depth, pts_feats, img_metas) # b n c z h w
                losses = dict()
                losses_pts,_ = self.forward_pts_train(uni_feats.squeeze(1), points, img, img_metas, img_depth, 0, batch_rays_list[0])
                losses_pts['rgb_loss_adj'] = losses_pts['rgb_loss']
                losses_pts['depth_loss_adj'] = losses_pts['depth_loss']
                losses.update(losses_pts)
                return losses
        else:
            pts_feats, img_feats, img_depth = self.extract_feat(
                points=points, img=img, img_metas=img_metas
            )
            uni_feats = self.view_transform(img_feats, img_depth, pts_feats, img_metas) # b n c z h w
            losses = dict()
            losses_pts = self.forward_pts_train(uni_feats.squeeze(1), points, img, img_metas, img_depth)
            losses.update(losses_pts)
            return losses
    
    @force_fp32()
    def shift_batch_feature(self, input, img_metas, adj_id, key_id):
        # n, c, h, w = input.shape
        n,c,z,h,w = input.shape
        lidar2ego = img_metas[0]['lidar2ego'][0]
        ego2global = img_metas[0]['ego2global'][0]
        keylidar2keyego = torch.tensor(img_metas[key_id]['lidar2ego'][0],dtype=input.dtype,device=input.device)
        sweeplidar2sweepego = torch.tensor(img_metas[adj_id]['lidar2ego'][0],dtype=input.dtype,device=input.device)
        keyego2global = torch.tensor(img_metas[key_id]['ego2global'][0],dtype=input.dtype,device=input.device)
        sweepego2global = torch.tensor(img_metas[adj_id]['ego2global'][0],dtype=input.dtype,device=input.device)

        global2keyego = keyego2global.inverse()
        keyego2keylidar = keylidar2keyego.inverse()
        keylidar2sweeplidar = (keyego2keylidar @ global2keyego @ sweepego2global @ sweeplidar2sweepego).inverse()

        xs = torch.linspace(
            0, w - 1, w, dtype=input.dtype,
            device=input.device).view(1, 1, w).expand(z, h, w)
        ys = torch.linspace(
            0, h - 1, h, dtype=input.dtype,
            device=input.device).view(1, h, 1).expand(z, h, w)
        zs = torch.linspace(
            0, z - 1, z, dtype=input.dtype,
            device=input.device).view(z, 1, 1).expand(z, h, w)
        voxel = torch.stack((xs, ys, zs, torch.ones_like(xs)), -1)
        voxel = voxel.view(1, z, h, w, 4).expand(n, z, h, w, 4).view(n, z, h, w, 4, 1)



        voxel = keylidar2sweeplidar.matmul(voxel)


        normalize_factor = torch.tensor([w - 1.0 ,h - 1.0, z - 1.0],
                                        dtype=input.dtype,
                                        device=input.device)
        voxel = voxel[:, :, :, :, :3, 0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
        output = F.grid_sample(input, voxel.to(input.dtype), align_corners=True)
        return output

    def shift_batch_voxel_feature(self, uni_feats, img_metas, key_id, bz):
        voxel_feat_list = []
        for adj_id in range(bz):
            if adj_id == key_id:
                continue
            voxel_feat = self.shift_batch_feature(uni_feats[adj_id],img_metas, adj_id, key_id) #(n, d, z, h, w)
            voxel_feat_list.append(voxel_feat)
        adj_voxel_feat = torch.cat(voxel_feat_list,dim=1)
        adj_voxel_feat = adj_voxel_feat.permute(0,2,3,4,1).contiguous()
        shape = adj_voxel_feat.shape[:-1]
        adj_voxel_feat = adj_voxel_feat.view(-1,adj_voxel_feat.shape[-1])
        adj_voxel_feat = self.voxel_fusion(adj_voxel_feat)
        adj_voxel_feat = adj_voxel_feat.view(*shape,-1).permute(0,4,1,2,3).contiguous()
        return adj_voxel_feat

    def forward_test(self, img_metas, points=None, img=None, **kwargs):
        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    "num of augmentations ({}) != num of image meta ({})".format(
                        len(points), len(img_metas)
                    )
                )

        assert num_augs == 1
        if not isinstance(img_metas[0], list):
            img_metas = [img_metas]
        if not isinstance(img, list):
            img = [img]
        results = self.simple_test(img_metas[0], points, img[0])

        return results

    def simple_test(self, img_metas, points=None, img=None):
        """Test function without augmentaiton."""
        pts_feat, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        l = 4
        uni_feats = self.view_transform(img_feats, img_depth, pts_feat, img_metas) # b n c z h w
        uni_feats = uni_feats[:,0]
        img, img_depth, img_metas = self.adjust_img_metas(img, img_depth, img_metas, keyid=0)
        batch_rays = self.pts_bbox_head.sample_rays_test(points, img, img_metas,l)
        results = self.pts_bbox_head(
            uni_feats, batch_rays, img_metas
        )
        return results
