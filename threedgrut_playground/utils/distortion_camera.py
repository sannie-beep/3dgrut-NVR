from kaolin.render.camera import Camera
import torch

# Extends the kaolin camera class to include distortion coefficients

class DistortionCamera(Camera):
    def __init__(self, *args, distortion_coefficients=None, **kwargs):
        """
        Initialises the DistortionCamera, which extends the kaolin Camera class to include distortion coefficients.
        Args:
            *args: Positional arguments for the Camera class.
            distortion_coefficients (list or None): Distortion coefficients to be used for the camera.
            **kwargs: Keyword arguments for the Camera class.
        """
        super().__init__(*args, **kwargs)
        self.distortion_coefficients = distortion_coefficients if distortion_coefficients is not None else None

    def set_distortion_coefficients_ds(self, distortion_coefficients):
        """
        Sets the distortion coefficients for the camera.
        Args:
            distortion_coefficients (list): List of distortion coefficients.
        """
        if len(distortion_coefficients) != 6:
            raise ValueError("Distortion coefficients must be a list of 6 values.")
        self.distortion_coefficients = distortion_coefficients

    def get_distortion_coefficients(self):
        """
        Returns the distortion coefficients of the camera.
        Returns:
            list or None: List of distortion coefficients, or None if no coefficients are set.
        """
        return self.distortion_coefficients
    
    @classmethod
    def from_args(cls, *args, distortion_coefficients=None, **kwargs):
        """
        Creates a DistortionCamera instance from the given arguments, passing all arguments to the superclass's from_args,
        and setting distortion_coefficients.
        """
        camera = super().from_args(*args, **kwargs)
        camera.distortion_coefficients = distortion_coefficients
        return camera
    
def wtf():
    """
    A placeholder function that does nothing.
    """
    pass