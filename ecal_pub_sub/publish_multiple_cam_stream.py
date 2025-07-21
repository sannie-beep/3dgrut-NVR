#!/usr/bin/env python3

import os
import sys
import time

import argparse
import numpy as np

#sys.path.insert(0, '/usr/lib/python3/dist-packages')
import ecal.core.core as ecal_core

sys.path.append('/opt/vilota/bin')
sys.path.append('/opt/vilota/python')


from capnp_publisher import CapnpPublisher
import capnp

sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()

import image_capnp as eCALImage
import threading, time, argparse, numpy as np, ecal.core.core as ecal_core

#!/usr/bin/env python3

import os
import sys
import time
import argparse
import threading

import numpy as np
import ecal.core.core as ecal_core
from capnp_publisher import CapnpPublisher
import capnp

# Setup Cap'n Proto import paths
sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()
import image_capnp as eCALImage


#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import ecal.core.core as ecal_core

from capnp_publisher import CapnpPublisher
import capnp

# Ensure Cap'n Proto imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '../vk_sdk/capnp'))
sys.path.append('/opt/vilota/messages')
capnp.add_import_hook()
import image_capnp as eCALImage


def create_publisher(cam_name: str, fps: int = 30) -> CapnpPublisher:
    """
    Worker process that publishes frames from one camera at a shared start time.
    """
    topic = f"S0/{cam_name.lower()}"
    pub = CapnpPublisher(topic, "Image")
    return pub

    

def load_frames_npz(cam_name: str = "CamX"):
    filepath = f'./{cam_name}.npz'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No frames found for camera {cam_name}. Please ensure the file {filepath} exists.")
    data = np.load(filepath)
    frames = data['frames']
    #print("Frames shape:", frames.shape)
    return frames



def build_image_message(bgr8_array : np.ndarray, index: int):
    """
    Builds a Capnp image message from a (h, w, 3) numpy array
    representing one single image in the sequence indexed by an int

    Args:
        bgra_array (np.ndarray): Image array of size H x W x 3 with BGR values from 0-255
        index (int): Integer index in the sequence
    """
    msg = eCALImage.Image.new_message()
    msg_header = msg.header
    msg_header.seq = index
    msg_header.stampMonotonic = index # * gap in nanoseconds ( inverse of framerate in nanoseconds)
    msg.encoding = eCALImage.Image.Encoding.bgr8
    msg.data = bgr8_array.tobytes()
    msg.height, msg.width = bgr8_array.shape[0], bgr8_array.shape[1]
    #print(type(msg))
    #print("Success!")
    return msg   

import time


def main():
    # frames_a = load_frames_npz("CamA")
    # frames_b = load_frames_npz("CamB")
    # frames_c = load_frames_npz("CamC")
    # frames_d = load_frames_npz("CamD")

    # print(f"A:{frames_a.shape}")
    # print(f"B:{frames_b.shape}")
    # print(f"C:{frames_c.shape}")
    # print(f"D:{frames_d.shape}")
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    ecal_core.initialize(sys.argv, "publish_multiple_cam_stream")
    ecal_core.set_process_state(1, 1, "Image Publisher Running")

    cams = ["CamA"]
    publishers = []
    all_frames = []
    # for cam in cams:
    publisher = create_publisher(cams[0])
    publishers.append(publisher)
    frames = load_frames_npz(cam_name = cams[0])
    all_frames.append(frames)

    ended = [False] *4
    num_frames = all_frames[0].shape[0]
    print(num_frames)

    cam_index = 0

    pub_index = 0
    while ecal_core.ok():
        num_cams = len(cams)
        
        for i in range(num_frames):
            # for cam_index in range(num_cams):
            # cam_pub = publishers[cam_index]
            cam_frames = all_frames[cam_index]
            msg = build_image_message(cam_frames[i], i)
            publisher.send(msg.to_bytes())
            time.sleep(1/30)
            if i == num_frames - 1:
                # ended[cam_index] = True
                break
        

            
        break    
        
            # print("Published all frames, exiting...")
        
        
   


if __name__ == "__main__":
    main()
