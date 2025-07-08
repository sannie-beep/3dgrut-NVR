from threedgrut_playground.utils.distortion_camera import DistortionCamera as Camera
import torch
import os
import json
from typing import List, Tuple, Union
import numpy as np

def view_matrix_from_rotation_translation(
            rotation: List,
            translation: List,
            offset: 'List[float]' = [1.0, 2.0, 3.0]
        ) -> np.ndarray:
        """
        Constructs a 4×4 view matrix from rotation and translation, overriding zero-rotation for Cam 0 if needed,
        and applying a global offset so all cameras shift together.
        """
        import numpy as np

        # Convert to arrays
    
        R = np.array(rotation, dtype=float).reshape(3,3)   # shape (3,3)
        t = np.array(translation, dtype=float).reshape(3,1)  # shape (3,1)

        # Override zero rotation to identity
        if np.allclose(R, 0):
            # This is likely Cam 0 with zero extrinsics; give it identity orientation
            R = np.eye(3, dtype=float)
            # Optionally: ensure t is zero so offset purely determines new center
            t = np.zeros((3,1), dtype=float)

        # Apply global offset: shift camera center by `offset` in world space
        offset_arr = np.array(offset, dtype=float).reshape(3,1)
        # Assuming rotation/translation denote world-to-camera extrinsic:
        t = t - R @ offset_arr

        # Build view matrix
        upper = np.hstack([R, t])               # shape (3,4)
        bottom = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=upper.dtype)
        view_matrix = np.vstack([upper, bottom])  # shape (4,4)
        return view_matrix

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

        # Calibration file folder
        self.calibration_folder = "./calibration_files/"  # Default folder for calibration files

        # Calibration file path
        self.calibration_path = "vkl.json"  # Default path to the calibration file

        # Helper for loading cameras from the calibration file
        self.camera_loader = CalibrationCameraLoader()

        # Distortion params for each camera
        self.camera_distortion_params = []

        self.v_translation = [0.0,0.0,0.0]
        self.v_rotation = [0.0,0.0,0.0]  # Yaw, pitch, roll in radians

        # Vilota device object ( with camera rig, name and serial no.)
        self.vilota_device = VDevice()

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
        path = self.calibration_folder + path  # Ensure the path is relative to the calibration folder
        if not os.path.exists(path):
            raise FileNotFoundError(f"Calibration file not found at {path}.")
        self.calibration_path = path
        
        return f"Calibration path set to {self.calibration_path}"
        
    def load_from_calibration(self):
        """
        Loads cameras from a Vilota calibration file and places them into the Vdevice
        """

        if self.calibration_path:
            calibraton_filepath = self.calibration_folder + self.calibration_path
        else: calibraton_filepath = self.calibration_folder + 'vkl.json'

        self.camera_loader.check_calibration_file_exists(calibraton_filepath)

        if not self.vilota_device:
            self.vilota_device = VDevice()

        for index in range(0,4):
            # Extract rig index and cam params
            rig_idx, cam_params = self.camera_loader.parse_cali_json_to_camera_params(calibraton_filepath, index)
            camera = self.camera_loader.cam_params_to_distortion_camera(cam_params, device='cpu')

            # Extract distortion params for double sphere.
            distortion_params_all = cam_params['distortionCoeff'] # grabs distortion coefficients from the camera parameters
            double_sphere_distortion_params = distortion_params_all[5:11]


            # Add camera to vilota device rig
            self.vilota_device.add_one_camera(rig_idx, camera, double_sphere_distortion_params)
        
        self.vilota_device.save_rig(initial = True)
        #yaw, pitch, roll = INITIAL_ROTATION
        #self.vilota_device.move_rig()

    
    def verbose_string(self):
        """
        String representation of the NVR
        """
        reply = f"Novel View Renderer with 1 Vilota device:\n{self.vilota_device.verbose_string()}\n"
        return reply
    
    def __str__(self):
        """
        Returns a string representation of the NovelViewRenderer.
        """
        return self.vilota_device.__str__()
    
    def get_num_cameras(self) -> int:
        """
        Returns the number of cameras in the renderer.
        """
        return self.vilota_device.get_num_cameras()
    
    def get_cameras_sorted(self) -> dict:
        """
        Returns the dictionary of cameras in the renderer.
        """
        cameras = self.vilota_device.get_cameras()
        # Sort cameras by their index
        sorted_cameras = dict(sorted(cameras.items()))
        return sorted_cameras
    
    def get_camera_by_index(self, index: int) -> Camera:
        """
        Returns a camera by its index.
        
        Args:
            index (int): The index of the camera to retrieve.
        
        Returns:
            Camera: The camera object corresponding to the given index.
        """
        cameras = self.get_cameras_sorted()
        if index in cameras:
            return cameras[index][0]
    
    def get_all_distortion_params(self) -> List[List[float]]:
        """
        Returns a list of all distortion parameters for the cameras in the VDevice.
        """
        return [distortion_params for _, distortion_params in self.get_cameras_sorted().values()]
    
    def move_vdevice(self):
        """
        Moves the Vilota device by self's set params
        """
        x = self.v_translation[0]
        y = self.v_translation[1]
        z = self.v_translation[2]
        y, p, r = self.v_rotation[0], self.v_rotation[1], self.v_rotation[2]
        yaw = torch.tensor([y], dtype=torch.float64, device=torch.device('cpu'))
        pitch = torch.tensor([p], dtype=torch.float64, device=torch.device('cpu'))
        roll = torch.tensor([r], dtype=torch.float64, device=torch.device('cpu'))
        # Move the Vilota device rig
        self.vilota_device.move_rig(x, y, z, yaw, pitch, roll)


    
    def get_distortion_of_cam(self, index: int) -> List[float]:
        """
        Returns the distortion parameters for a specific camera by its index.
        
        Args:
            index (int): The index of the camera to get distortion parameters for.
        
        Returns:
            List[float]: The distortion parameters for the specified camera.
        """
        cameras = self.vilota_device.get_cameras()
        if index in cameras:
            return cameras[index][1]
    
class VDevice:
    """
    A class representing the simulated camera rig to represent the Vilota device.
    This class if used to simulate the device's camera rig and its properties
    """
    def __init__ (self):
        """
        Initializes the simulated VDevice with a list of cameras and their distortion parameters.
        """
        self.cameras = {} # A dictionary with key: camera id, value: Tuple of (Camera, distortion parameters)
        self.origin_camera = None  # The origin camera from which all other camera extrinsics are defined
        self.name = "Vilota Device"
        self.serial_no = "No serial number assigned yet."
        self.saved_rigs = {}
        self.num_saved_rigs = 1
        self.device = torch.device('cpu')  # Default device is CPU, can be changed later

    def set_name_and_serial_number(self, name: str, serial_no: str):
        self.name = name
        self.serial_no = serial_no

    def get_num_cameras(self):
        return len(self.cameras)
    
    def get_cameras(self):
        """
        Returns the dict of cameras in the VDevice.
        """
        return self.cameras
    
    def __str__(self):
        """
        Returns a string representation of the VDevice.
        """
        return f"Name: {self.name}, Serial No.: {self.serial_no}"



    def add_one_camera(self, idx: int, camera: Camera, distortion_params: List[float]):
        """
        Adds one camera to the VDevice's camera list

        Args:
        idx (int): The camera's index in the calibration file (NOT the same as order, it is to do with the rig order)
        camera: A kaolin.render.camera.Camera object representing the camera to be added.
        distortion_params: The list of distortion parameters corresponding to this camera
        """
        self.cameras[idx] = (camera, distortion_params)
        if idx == 0:
            self.origin_camera = (idx, camera)
    
    def verbose_string(self) -> str:
        """
        String representation of the VDevice class
        """
        return f"{self.list_cameras()} \n Main camera: {'Set successfully' if self.origin_camera else 'No main registered'}"
    
    def list_cameras(self) -> str:
        """
        Lists all cameras in the VDevice.
        """
        reply = f"Name: {self.name}\nSerial no.: {self.serial_no}\nWith {len(self.cameras)} cameras:\n"
        for idx, (camera, distortion_params) in self.cameras.items():
            reply += f"Camera {idx}: {camera.named_params()}\n"
        
        return reply
    

    def save_rig(self, initial = False):
        """
        Saves the current rotation and translation settings 

        Args:
        initial (bool): if this is the first save of the rig, it will save it with special
        dict key "Initial", to save the rig separate from worldspace wuth orgin_camera as the origin
        extrinsics
        """
        # if self.origin_camera is None:
        #     raise ValueError("Origin camera has not been set. Cannot save rig.")
        
        # Save the original camera parameters
        if initial:
            self.saved_rigs["Initial"] = self.cameras
        # Else save the current state of the cameras
        else:
            self.saved_rigs[f"Orientation {self.num_saved_rigs}"] = self.cameras
            self.num_saved_rigs += 1
    
    def set_rig_position(
            self,
            x: float = 0.0,
            y: float = 0.0,
            z: float = 0.0,
            yaw = [0.0, 0.0, 0.0],
            pitch = [0.0, 0.0, 0.0],
            roll = [0.0, 0.0, 0.0]
    ):
        """
        Sets the camera rig to the given position by constructing the view matrix and using the kaolin
        camera update function
        """
        view_mat = view_matrix_from_rotation_translation([yaw,pitch,roll], [x,y,z], offset = [0,0,0])
        print(view_mat)



    def move_rig(
        self,
        x: float = 0.50,
        y: float = 0.50,
        z: float = 0.50,
        yaw: torch.tensor = torch.tensor([0.50], dtype=torch.float64, device= torch.device('cpu')),
        pitch: torch.tensor = torch.tensor([0.50], dtype=torch.float64, device= torch.device('cpu')),
        roll: torch.tensor = torch.tensor([0.50], dtype=torch.float64, device= torch.device('cpu'))
    ):
        """
        Moves the camera rig by the given translation and rotation. All cameras'
        translations and rotations will be updated accordingly
        
        Args:
            x (float): Translation along the x-axis.
            y (float): Translation along the y-axis.
            z (float): Translation along the z-axis.
            yaw (float): Rotation around the z-axis in radians.
            pitch (float): Rotation around the y-axis in radians.
            roll (float): Rotation around the x-axis in radians.
        """
        # if self.origin_camera is None:
        #     raise ValueError("Origin camera is not set. Cannot move camera rig.")
        
        
        
        for idx, (camera, distortion_params) in self.cameras.items():
            # Debug: print the current camera params
            #print(f"Cam {idx} before move: {camera.named_params()}")
            # Update each camera's extrinsic by translating and rotating
            camera.extrinsics.translate(torch.tensor([x,y,z], dtype=torch.float64, device=camera.device))
            camera.extrinsics.rotate(yaw, pitch, roll)
            # Debug: print the updated camera params
            #print(f"Cam {idx} after move: {camera.named_params()}")
            
           

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
    

    



    def parse_cali_json_to_camera_params(
            self,
            file_path: str, 
            index: int = 0          
        ) -> Tuple[int, dict]:
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
        
        cam_rig_index = data.get('cameraData', {})[index][0]
        camera_data = data.get('cameraData', {})[index][1]

        translation = camera_data.get('extrinsics', {}).get('translation', [])
        trans_vector = list(translation.values())
        cam_type = camera_data.get('cameraType', 9)
        if cam_type != 0:
            has_distortion = False
            print(f"Camera {index} is not a distortion camera, skipping distortion params.")
        else:
            has_distortion = True
        
        camera_params = {
            'cameraType': camera_data.get('cameraType', 'unknown'),
            'distortionCoeff': camera_data.get('distortionCoeff', []),
            'ext_rotation' : camera_data.get('extrinsics', {}).get('rotationMatrix', []),
            'ext_translation' : trans_vector,
            'intrinsics': camera_data.get('intrinsicMatrix', {}),
            'width': camera_data.get('width', 1920),
            'height': camera_data.get('height', 1200),
            'hasDistortion': has_distortion
        }

        return cam_rig_index, camera_params

    def cam_params_to_distortion_camera(
            self,
            camera_params: dict, 
            device: Union[torch.device, str] = 'cpu'
    ) -> Camera:
        """
        Converts loaded camera parameters to a Distortion Camera object.

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
        print("ck1")
        rotation_matrix = camera_params['ext_rotation']
        if rotation_matrix is None or len(rotation_matrix) == 0:
            rotation_matrix = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

        #print(f"Rotation Matrix: {rotation_matrix}")
        translation_vector = camera_params['ext_translation']
        #print(f"Translation Vector: {translation_vector}")
        view_matrix = view_matrix_from_rotation_translation(rotation_matrix, translation_vector)

        print(f"Cam has distortion? {camera_params['hasDistortion']}")
        # Convert to Distortion Camera object
        distortion_camera = Camera.from_args(
            view_matrix = torch.tensor(view_matrix, dtype=torch.float64, device=device),
            focal_x = f_x,
            focal_y = f_y,
            x0 = c_x,
            y0 = c_y,
            width = camera_params['width'],
            height = camera_params['height'],
            distortion_coefficients = camera_params['distortionCoeff'] if camera_params['hasDistortion'] else None,
            dtype=torch.float64,
            device = device
        )
        if distortion_camera.__getattribute__("distortion_coefficients") is not None:
            print(f"Yay distortion!: {distortion_camera.distortion_coefficients[0]}")
        return distortion_camera

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
    


def test_novel_view_renderer():
    """
    Test function to demonstrate the usage of NovelViewRenderer.
    """
    renderer = None  # Replace with actual renderer instance
    nvr = NovelViewRenderer(renderer)
    nvr.calibration_folder = "~/3dgrut/calibration/"
    # Set calibration path
    nvr.set_calibration_path('vkl.json')
    
    # Load cameras from calibration file
    nvr.load_from_calibration()
    nvr.vilota_device.list_cameras()
    #print(nvr)

    
    #print(f"Params: {nvr.camera_distortion_params}")

#test_novel_view_renderer()