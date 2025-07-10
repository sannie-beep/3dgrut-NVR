# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import polyscope as ps
from typing import List, Union
#from kaolin.render.camera import Camera
from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera

"""
This module is to be included in next version of kaolin 0.18.0.
As of March 26, 2025 the latest public release is kaolin 0.17.0, hence it's included here independently.
"""

def polyscope_to_kaolin_camera(
    ps_camera: ps.core.CameraParameters,
    width: int,
    height: int,
    near: float = 1e-2,
    far: float = 1e2,
    fx = None,
    fy = None,
    cx = None,
    cy = None,
    is_vilota: bool = False,
    distortion_coefficients: List[float] = None,
    device: Union[torch.device, str] = 'cpu'
) -> Camera:
    """ Converts a polyscope camera (polyscope.core.CameraParameters) to kaolin Camera format (kaolin.render.camera.Camera).
    The converted information includes the camera extrinsics, the image plane dimensions and field of view.
    Additional parameters that kaolin cameras assume, such as near, far plane and device
    can be passed explicitly if needed.

    Args:
        ps_camera (ps.core.CameraParameters): A polyscope camera object.
        is_novel_view (bool): If True, the camera will be rendered as a distortion camera
        width (int): Image plane width in pixels.
        height (int): Image plane height in pixels.
        near (optional, float): near clipping plane, defines the min depth of the view frustrum.
        far (optional, float): far clipping plane, define the max depth of the view frustrum.
        distortion_coefficients (optional, List[float]): Distortion coefficients for the camera. If loaded from calibration file, the distortion
        camera will have a list of 6 coefficients, which are used to apply double sphere unprojection for ray generation.
            If None, no distortion is applied. Default: None
        device (optional, torch.device or str): the device on which camera parameters will be allocated. Default: cpu
    Returns:
        (kaolin.render.camera.Camera):
            A kaolin camera object.
    """
    view_matrix = ps_camera.get_view_mat()
    fov_y = ps_camera.get_fov_vertical_deg() * np.pi / 180.0    # to radians
    cam =  Camera.from_args(
        view_matrix=view_matrix,
        fov=fov_y,
        width=width, height=height,
        near=near, far=far,
        dtype=torch.float64,
        device=device,
        distortion_coefficients= distortion_coefficients
    )
    if is_vilota:
        cam = Camera.from_args(
            view_matrix=view_matrix,
            width=width, height=height,
            focal_x=fx, focal_y=fy,
            x0=cx, y0=cy,
            dtype=torch.float64,
            device=device,
            distortion_coefficients=distortion_coefficients,
            intrinsic_params=[fx, fy, cx, cy]
        )
    
    return cam



def polyscope_from_kaolin_camera(camera: Camera) -> ps.core.CameraParameters:
    """ Converts a kaolin camera (kaolin.render.camera.Camera) to a polyscope camera format (polyscope.core.CameraParameters).
    polyscope cameras are always assumed to exist on a cpu device.
    The converted information includes the camera extrinsics, and intrinsics for the field of view.

    Args:
        camera (kaolin.render.camera.Camera): A polyscope camera object.
    Returns:
        (ps.core.CameraParameters):
            A polyscope camera object.
    """
    view_matrix = camera.view_matrix()
    ps_cam_param = ps.CameraParameters(
        ps.CameraIntrinsics(fov_vertical_deg=camera.fov_y.detach().cpu().numpy(), aspect=camera.width / camera.height),
        ps.CameraExtrinsics(mat=view_matrix[0].detach().cpu().numpy())
    )
    return ps_cam_param
