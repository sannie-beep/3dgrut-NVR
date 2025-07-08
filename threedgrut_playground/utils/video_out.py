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

from __future__ import annotations
import json
import os
from typing import List, Tuple, Union
import numpy as np
import torch
import cv2
from tqdm import tqdm
from scipy.interpolate import splprep, splev
from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera
from threedgrut.utils.logger import logger
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.model.background import BackgroundColor
from threedgrut_playground.utils.mesh_io import load_mesh, load_materials, load_missing_material_info, create_procedural_mesh
from threedgrut_playground.utils.depth_of_field import DepthOfField
from threedgrut_playground.utils.spp import SPP
from threedgrut_playground.tracer import Tracer
from threedgrut_playground.utils.kaolin_future.transform import ObjectTransform
from threedgrut_playground.utils.kaolin_future.conversions import polyscope_from_kaolin_camera, polyscope_to_kaolin_camera
from threedgrut_playground.utils.kaolin_future.fisheye import generate_fisheye_rays
from threedgrut_playground.utils.kaolin_future.interpolated_cameras import camera_path_generator


class VideoRecorder:
    """ A module for saving video footage of camera trajectories within the Playground"""

    MODES = ['path_smooth', 'path_spline', 'cyclic', 'depth_of_field']

    def __init__(self,
        renderer,
        trajectory_output_path="output.mp4",
        cameras_save_path="cameras.npy",
        mode='path_smooth',
        frames_between_cameras=60,
        video_fps=30,
        min_dof=2.5,
        max_dof=24
    ):
        """
        Creates a video recorder for saving camera animation along trajectories.
        Args:
            renderer: Reference to the Playground rendering engine, interfaced through a 'render()' function.
            trajectory_output_path (str): Output path for saved video
            cameras_save_path (str): Save path for storing and loading camera trajectories.
            mode (str): Determines how keyframes are interpolated:
                'path_smooth' - shifts the camera in a path along the trajectory, using smoothstep interpolation
                'path_spline' - shifts the camera in a path along the trajectory, using catmull-rom splines
                'cyclic' - shifts the camera in a cyclic trajectory, determined by fitting a bspline
                'depth_of_field' - interpolates the depth of field over a static frame (first frame in trajectory)
            frames_between_cameras (int): How many cameras are inserted between keyframe cameras in the trajectory
            video_fps (int): FPS of exported video
            min_dof (float): For 'depth_of_field' mode only, determines the minimum dof for interpolation start / end
            max_dof (float): For 'depth_of_field' mode only, determines the maximum dof for interpolation start / end
        """
        self.renderer = renderer

        # List of kaolin.render.camera.Camera objects forming the trajectory
        self.trajectory = []

        # Selected interpolation modes
        self.mode = mode

        # Output path for generated video file
        self.trajectory_output_path = trajectory_output_path

        # For saving and loading trajectory paths
        self.cameras_save_path = cameras_save_path

        # Camera FPS -- how many cameras are interpolated between key views
        self.frames_between_cameras = frames_between_cameras

        # Saved Video FPS
        self.video_fps = video_fps

        # For depth-of-field interpolation mode only - determines the boundary
        self.min_dof = min_dof
        self.max_dof = max_dof

    def add_camera(self, camera: Camera):
        self.trajectory.append(camera)

    def reset_trajectory(self):
        self.trajectory = []

    def save_trajectory(self):
        torch.save(self.trajectory, self.cameras_save_path)

    def load_trajectory(self):
        self.trajectory = torch.load(self.cameras_save_path)

    def render_dof_trajectory(self):
        out_video = None
        old_use_dof = self.renderer.use_depth_of_field
        old_focus_z = self.renderer.depth_of_field.focus_z
        try:
            camera = self.trajectory[0]
            dofs = np.linspace(self.min_dof, self.max_dof, self.frames_between_cameras)
            for dof in tqdm(dofs):
                self.renderer.use_depth_of_field = True
                self.renderer.depth_of_field.focus_z = dof
                rgb = self.renderer.render(camera)['rgb']

                if out_video is None:
                    out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                                self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
                data = rgb[0].clip(0, 1).detach().cpu().numpy()
                data = (data * 255).astype(np.uint8)
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                out_video.write(data)
        finally:
            self.renderer.use_depth_of_field = old_use_dof
            self.renderer.depth_of_field.focus_z = old_focus_z

    def render_linear_trajectory(self, interpolation_mode='polynomial'):
        if len(self.trajectory) < 2:
            raise ValueError('Rendering a path trajectory requires at least 2 cameras.')
        elif interpolation_mode == 'catmull_rom' and len(self.trajectory) < 4:
            raise ValueError('Rendering a path with a spline interpolated trajectory requires at least 4 cameras.')

        out_video = None
        distortions = self.trajectory[0].distortion_coefficients if hasattr(self.trajectory[0], 'distortion_coefficients') else None
        interpolated_path = camera_path_generator(
            trajectory=self.trajectory,
            frames_between_cameras=self.frames_between_cameras,
            interpolation=interpolation_mode
        )

        for camera in tqdm(interpolated_path):
            if not hasattr(camera, 'distortion_coefficients') or camera.distortion_coefficients is None:
                # give it the attribute
                camera.distortion_coefficients = distortions
        
            else:
                print(f"Camera {camera} has distortion coefficients: {camera.distortion_coefficients}")
            
            rgb = self.renderer.render(camera)['rgb']
            #check if the camera has distortion coefficients
            

            if out_video is None:
                out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)
        out_video.release()

    def render_continuous_trajectory(self):
        if len(self.trajectory) < 4:
            raise ValueError('Rendering a continuous trajectory requires at least 4 cameras.')

        def _fit_bspline(data_points, frames_between_cameras):
            stacked_data_points = torch.stack([dp for dp in data_points], dim=1).cpu().numpy()
            tck, u = splprep(stacked_data_points, u=None, s=0.0, per=1)
            u_new = np.linspace(u.min(), u.max(), frames_between_cameras * len(data_points))
            interpolated_cam_param = np.stack(splev(u_new, tck, der=0)).T
            interpolated_cam_param = torch.tensor(interpolated_cam_param)
            return interpolated_cam_param

        cam_poses = _fit_bspline(
            data_points=[c.cam_pos().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ats = _fit_bspline(
            data_points=[c.cam_pos().squeeze() - c.cam_forward().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )
        cam_ups = _fit_bspline(
            data_points=[c.cam_up().squeeze() for c in self.trajectory],
            frames_between_cameras=self.frames_between_cameras
        )

        first_cam = self.trajectory[0]
        interpolated_path = []
        for eye, at, up in tqdm(zip(cam_poses, cam_ats, cam_ups)):
            interpolated_path.append(
                Camera.from_args(
                    eye=eye,
                    at=at,
                    up=up,
                    fov=first_cam.fov(in_degrees=False),
                    width=first_cam.width, height=first_cam.height,
                    device=first_cam.device
                )
            )

        out_video = None
        for camera in tqdm(interpolated_path):
            rgb = self.renderer.render(camera)['rgb']
            if not hasattr(camera, 'distortion_coefficients') or camera.distortion_coefficients is None:
                ValueError(
                    f"Camera {camera} does not have distortion coefficients. "
                    "Ensure all cameras in the trajectory have valid distortion coefficients."
                )
            else: 
                print(f"Camera {camera} has distortion coefficients: {camera.distortion_coefficients}") 
            if out_video is None:
                out_video = cv2.VideoWriter(self.trajectory_output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.video_fps, (rgb.shape[2], rgb.shape[1]), True)
            data = rgb[0].clip(0, 1).detach().cpu().numpy()
            data = (data * 255).astype(np.uint8)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            out_video.write(data)
        out_video.release()

    def render_video(self):
        """
        Renders the trajectory to a video file, according to the set mode (see init()).
        """
        if self.mode == 'depth_of_field':
            self.render_dof_trajectory()
        elif self.mode == 'cyclic':
            self.render_continuous_trajectory()
        elif self.mode == 'path_smooth':
            self.render_linear_trajectory('polynomial')
        elif self.mode == 'path_spline':
            self.render_linear_trajectory('catmull_rom')
        else:
            raise ValueError(f'Unknown mode: {self.mode}')

class NovelViewRenderer: 
    """ A class for rendering novel views of the scene from a Vilota calibration file"""
    def __init__ (
        self,
        renderer
    ):
        """
        Initializes the renderer with a reference to the Playground rendering engine.
        Args:
            renderer: Reference to the Playground rendering engine, interfaced through a 'render()' function.
        """
        self.renderer = renderer
        
        # List of cameras to render from 
        self.cameras = []

        # Calibration file path
        self.calibration_path = "./calibration_sample.json"  # Default path to the calibration file

        # Helper for loading cameras from the calibration file
        self.camera_loader = CalibrationCameraLoader()

        # Distortion params for each camera
        self.camera_distortion_params = []

        # Number of cameras in this calibration
        self.num_cameras = 4
    
    # TODO: remove this once the GUI button is fixed
    def set_num_cameras(self, num_cameras: int):
        """
        Sets the number of cameras to render from. To prevent reloading more cameras than needed.
        This is used to check if the load button was pressed multiple times.
        Args:
            num_cameras (int): Number of cameras to render from.
        """
        if num_cameras:
            self.num_cameras = num_cameras
        pass


    def reset(self):
        """
        Resets the renderer by clearing the list of cameras and camera distortion parameters.
        """
        self.cameras = []
        self.camera_distortion_params = []
    

    def add_camera(self, camera: Camera):
        """
        Private function that adds a camera to the list of cameras to render from.
        Args:
            camera: A kaolin.render.camera.Camera object representing the camera to be added.
        """
        self.cameras.append(camera)

    def set_calibration_path(self, path: str):
        """
        Sets the path to the Vilota calibration file.
        Args:
            path (str): Path to the calibration file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found at {path}.")
        self.calibration_path = path
        
        return f"Calibration path set to {self.calibration_path}"
        
    def load_from_calibration(self):
        """
        Loads cameras from a Vilota calibration file.
        """

        if self.calibration_path:
            calibraton_filepath = self.calibration_path
        else: calibraton_filepath = './calibration_sample.json'

        self.camera_loader.check_calibration_file_exists(calibraton_filepath)

        # Parse the json calibration file and extract cameras and corresponding distortion parameters from it.
        for index in range(0,4): # loop through the cameras in the calibration file, up to 4 cameras
            # Load camera parameters from the calibration file
            cam_params = self.camera_loader.parse_cali_json_to_camera_params(calibraton_filepath, index) # reads the calibration file and returns camera parameters
            kaolin_camera = self.camera_loader.cam_params_to_kaolin_camera(cam_params, device='cpu') # creates a kaolin camera object for that camera in the json
            print(f"Camera {index} parameters: {kaolin_camera.named_params()}") # print the camera parameters for debugging
            # Extract distortions from params
            distortion_params_all = cam_params['distortionCoeff'] # grabs distortion coefficients from the camera parameters

            double_sphere_distortion_params = distortion_params_all[5:11] # only take the slice needed for double spheres raygen function
            self.camera_distortion_params.append(double_sphere_distortion_params) # adds this camera's distortion params

            self.cameras.append(kaolin_camera) # adds the camera to the list of cameras to render from


    

class CalibrationCameraLoader:
    """ A helper class that loads a Kaolin camera from a Vilota calibration file"""
    def __init__(self):
        pass

    def check_calibration_file_exists(self, path):
        """
        Checks if calibration file.json exists at the given path.
        """
        if not path.endswith('.json'):
            raise ValueError("Calibration file must be a .json file.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found at {path}.")
        return True
    
    
    def view_matrix_from_rotation_translation(self, rotation, translation) -> List:
        """
        Helper function that constructs a view matrix from rotation and translation vectors.

        Args:
            rotation (list): A 3x3 rotation matrix.
            translation (list): A translation vector of length 3.

        Returns:
            list: A 4x4 view matrix.
        """
        R = rotation
        t = translation
        R = np.array(R)              # ensure it’s an array
        t = np.array(t)              # shape might be (3,)
        t = t.reshape(3, 1)          # force into column vector (3×1)

        upper = np.hstack([R, t])    # shape (3,4)
        bottom = np.array([[0, 0, 0, 1]], dtype=upper.dtype)
        view_matrix = np.vstack([upper, bottom])  # shape (4,4)
        return view_matrix


    def parse_cali_json_to_camera_params(
            self,
            file_path: str, 
            index: int = 0          
        ) -> dict:
        """
        Parses a .json calibration file and returns the camera parameters.

        Args:
            index (int): Index of the camera to parse from the calibration file. Default is 0.
            file_path (str): Path to the calibration.json file.

        Returns:
            dict: A dictionary containing the camera parameters.
        """
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        camera_data = data.get('cameraData', {})[index][1]
        translation = camera_data.get('extrinsics', {}).get('translation', [])
        trans_vector = list(translation.values())
        
        camera_params = {
            'cameraType': camera_data.get('cameraType', 'unknown'),
            'distortionCoeff': camera_data.get('distortionCoeff', []),
            'ext_rotation' : camera_data.get('extrinsics', {}).get('rotationMatrix', []),
            'ext_translation' : trans_vector,
            'intrinsics': camera_data.get('intrinsicMatrix', {}),
            'width': camera_data.get('width', 1920),
            'height': camera_data.get('height', 1200),
        }
        return camera_params


    def cam_params_to_kaolin_camera(
            self,
            camera_params: dict, 
            device: Union[torch.device, str] = 'cpu'
    ) -> Camera:
        """
        Converts loaded camera parameters to a Kaolin Camera object.

        Args:
            camera_params (dict): Dictionary containing camera parameters.
            device (optional, torch.device or str): the device on which camera parameters will be allocated. Default: cpu
        Returns:
            Camera: A Kaolin Camera object initialized with the provided parameters.
        """
        # Extract intrinsic params
        intrinsic_matrix = camera_params['intrinsics']
        f_x = intrinsic_matrix[0][0]
        f_y = intrinsic_matrix[1][1]
        c_x = intrinsic_matrix[0][2]
        c_y = intrinsic_matrix[1][2]

        # Extract extrinsic params
        rotation_matrix = camera_params['ext_rotation']
        #print(f"Rotation Matrix: {rotation_matrix}")
        translation_vector = camera_params['ext_translation']
        #print(f"Translation Vector: {translation_vector}")
        view_matrix = self.view_matrix_from_rotation_translation(rotation_matrix, translation_vector)

        # Convert to Kaolin Camera object
        kaolin_camera = Camera.from_args(
            view_matrix = torch.tensor(view_matrix, dtype=torch.float64, device=device),
            focal_x = f_x,
            focal_y = f_y,
            x0 = c_x,
            y0 = c_y,
            width = camera_params['width'],
            height = camera_params['height'],
            dtype=torch.float64,
            device = device
        )
        return kaolin_camera

    def get_sample_kaolin_camera(self) -> Tuple[List, Camera]:
        file_path = './calibration_sample.json'  # Replace with your .cali file path
        camera_params = self.parse_cali_json_to_camera_params(file_path)
        print(f"Camera width: {camera_params['width']}, height: {camera_params['height']}")
        
        # Convert to Kaolin Camera object
        kaolin_camera = self.cam_params_to_kaolin_camera(camera_params, device='cpu')

        # Grab distortion params
        distortion_coeffs = camera_params['distortionCoeff']
        distortion_coeffs = distortion_coeffs[5:11]
        print(f"Distortion Coefficients: {distortion_coeffs}")

        return distortion_coeffs, kaolin_camera
    