"""
Project: HYper-reduced Basis Reduction via Interactive Decomposition

@author: Esmaeil Rezaei
link: https://github.com/ishmaelrezaei/HYBRID

Note: This is a part of the HYBRID code the comprehensive code will
be available after we publish our paper.

To run this code with different examples, you can easily upload your
video to the 'VideoSample' folder and replace 'Sample1.mp4' with the filename
of your video."""

import sys
import cv2
import numpy as np
sys.path.insert(0, '../RBMs_Functions/')
from RBMs_Functions import RBMs
sys.path.pop(0)

if __name__ == "__main__":
    # Open the video sample file
    video_path = 'VideoSample/Sample1.mp4'  # Replace with the path to your video file
    vidcap = cv2.VideoCapture(video_path)
    # Get the total number of frames
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define a reduction level
    dMax = 400

    # Get the dimensions of the frames
    _, frame = vidcap.read()
    (nr, nc, nd) = frame.shape

    # Reset to the beginning of the video
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    Reconstructed_frame = np.zeros((nr, nc, nd), dtype=float)

    # Define the video file name and codec
    video_name = 'ReconstructedVideo.avi'

    height, width, layers = frame.shape

    # Create a VideoWriter object to write the video
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for f_idx in np.arange(total_frames):
        # Go through each frame
        print("Reading frame " + str(f_idx+1) + " from " + str(total_frames))
        # Read the frames
        _, frame = vidcap.read()
        frame = np.array(frame, dtype=float)
        # Go through each dimension of the frame
        for i in np.arange(nd):
            data_i = frame[:, :, i]
            # Model creation and run RBD
            RB = RBMs(data=data_i, tol=1e-6, col=dMax)
            bases, TransMat = RB.RBD()
            Reconstructed_frame[:, :, i] = np.dot(bases, TransMat)

        # Add each image as a frame to the video
        video.write(Reconstructed_frame.astype(np.uint8))
    # Release the VideoWriter when done
    video.release()
