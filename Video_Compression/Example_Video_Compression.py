# Image compression
import sys
import cv2
import numpy as np
sys.path.insert(0, '../RBMs_Functions/')
from RBMs_Functions import RBMs
sys.path.pop(0)

if __name__ == "__main__":
    # Open the video file
    video_path = 'VideoSample/Sample1.mp4'  # Replace with the path to your video file
    vidcap = cv2.VideoCapture(video_path)
    # Get the total number of frames
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    dMax = 200;
    _, frame = vidcap.read()

    (nr, nc, nd) = frame.shape;
    # Reset to the beginning of the video
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    Reconstructed_image = np.zeros((nr, nc, nd), dtype=float);


    # Define the video file name and codec
    video_name = 'video.avi'

    height, width, layers = frame.shape

    # Create a VideoWriter object to write the video
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for f_idx in np.arange(total_frames):
        print("Reading frame " + str(f_idx+1) + " from " + str(total_frames))
        # Read the frames
        ret, frame = vidcap.read()
        frame = np.array(frame, dtype=float)
        for i in np.arange(nd):
            data_i = frame[:, :, i];
            RB = RBMs(data=data_i, tol=1e-6, col=dMax)
            bases, TransMat = RB.RBD();
            Reconstructed_image[:, :, i] = np.dot(bases, TransMat);

        # Add each image as a frame to the video
        video.write(Reconstructed_image.astype(np.uint8))
    # Release the VideoWriter when done
    video.release()
