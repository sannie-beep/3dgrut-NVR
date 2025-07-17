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


parser = argparse.ArgumentParser(description="Publish image stream from a sequence of images.")
parser.add_argument('--cam_name', type=str, required=True, help='Camera name to load frames from (default: CamX) ' \
'Enter CamA, CamB, CamC or CamD, based on what .npz file you want to load from.')
args = parser.parse_args()

def load_sample_image():
    # if there's a file called bgr_image.npy, load it
    if os.path.exists('./bgr_image.npy'):
        sample_img = np.load('./bgr_image.npy')
    else:
        sample_img = np.array([[[255,255,255],[0,0,0],[255,255,255]],
                         [[0,0,0],[255,255,255],[0,0,0]],
                         [[255,255,255],[0,0,0],[255,255,255]]])
    return sample_img.astype(np.uint8)

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
    msg_header.stampMonotonic = index
    msg.encoding = eCALImage.Image.Encoding.bgr8
    msg.data = bgr8_array.tobytes()
    msg.height, msg.width = bgr8_array.shape[0], bgr8_array.shape[1]
    #print(type(msg))
    #print("Success!")
    return msg

def register_camera_name(cam_name: str):
    """
    Registers the camera name to initialise the publisher.
    """
    if not cam_name.startswith("Cam"):
        raise ValueError(f"Invalid camera name: {cam_name}. It should start with 'Cam'.")
    topic_name = "S0/" + cam_name.lower()
    return topic_name, cam_name


def main():
    sample_img= load_sample_image()
    msg = build_image_message(sample_img, 0)
    # print(msg.width, msg.height, msg.encoding)
    # print(msg.data)
    print("eCAL {} ({})\n".format(ecal_core.getversion(), ecal_core.getdate()))
    
    ecal_core.initialize(sys.argv, "publish_img_stream")
    ecal_core.set_process_state(1, 1, "Image Publisher Running")
    args = parser.parse_args()
    topic_name, cam_name = register_camera_name(args.cam_name)
    pub = CapnpPublisher(topic_name, "Image")

    #pub = CapnpPublisher("S0/camc", "Image")

    seq = 0
    frames = load_frames_npz(cam_name="CamB")
    num_frames = frames.shape[0]
    ended = False
    while ecal_core.ok():
        
        for i, frame in enumerate(frames):
            #print(f"Publishing frame {i}")
            msg = build_image_message(frame, i)
            pub.send(msg.to_bytes())
            if i == num_frames - 1:
                ended = True
                break
            
        if ended:
            print("Published all frames, exiting...")
            break
    
       
           
        
    #     time.sleep(0.01)  # 100 Hz

    ecal_core.finalize()

if __name__ == "__main__":
     main()

