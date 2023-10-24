# [HYper-reduced Basis Reduction via Interactive Decomposition](https://ishmaelrezaei.github.io/HYBRID/) in Python

> **Note:** This is a part of the HYBRID code; the comprehensive code will be available after we publish our paper.
> **Note:** 

In this project, we developed a high-speed dimensional reduction algorithm designed for efficiently handling massive matrices. Our work involved enhancing the Reduced Basis Dimension (RBD) algorithm by incorporating numerical solutions for partial differential equations. We are currently preparing to release the code for the HYBRID algorithm following its paper publication.

In the meantime, we offer access to the RBD algorithm, complete with two practical examples: image compression and video compression. It's important to note that the primary goal isn't limited to the application of HYBRID or RBD in video/image compression; instead, it focuses on using images and videos to effectively illustrate the efficiency of these algorithms.

![Lena](https://github.com/ishmaelrezaei/HYBRID/blob/main/Image_Compression/ReconstructedImage.png)

To see more detail on RBD please refere to:

  - Chen, Yanlai. "[Reduced basis decomposition: A certified and fast lossy data compression algorithm](https://www.sciencedirect.com/science/article/pii/S0898122115004630)." Computers & Mathematics with Applications (2015).

## Citation

    @article{chen2015reduced,
      title={Reduced basis decomposition: a certified and fast lossy data compression algorithm},
      author={Chen, Yanlai},
      journal={Computers \& Mathematics with Applications},
      volume={70},
      number={10},
      pages={2566--2574},
      year={2015},
      publisher={Elsevier}
      year={2017}
    }


## Example

To run any example, you can copy your image in the "Images" folder and your video in the "VideoSample" folder. Then, replace the file name in the Example_Image_Compression.py file for the image compression example and in the Example_Video_Compression.py file for the video example. Each example, uses RBD code which is in the class RBMs (RBMs_Functions folder).

    # Image compression
    file_name = 'Lena.png';
    
    # Video compression
    video_path = 'VideoSample/Sample1.mp4'

## Installing Dependencies

    pip install matplotlib
    pip install cv2
    pip install numpy
    pip install PIL
    pip install sys
    
