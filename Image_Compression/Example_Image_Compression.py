"""
Project: HYper-reduced Basis Reduction via Interactive Decomposition

@author: Esmaeil Rezaei
link: https://github.com/ishmaelrezaei/HYBRID

Note: This is a part of the HYBRID code the comprehensive code will
be available after we publish our paper.

To run this code with different examples, you can easily upload your
image to the 'Images' folder and replace 'Lena.png' with the filename
of your image.  If  you  intend to run  this code with a  dataset of
numbers, you can read your file  and import it, then place it in the
'thispic' variable in  the first lines.  Of course, you may need  to
write  some  code for visualizing errors when working with non-image
data. """

# Image compression
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
sys.path.insert(0, '../RBMs_Functions/')
from RBMs_Functions import RBMs
sys.path.pop(0)

if __name__ == "__main__":
    # Read the data (image)
    file_name = 'Lena.png'
    thispic = Image.open('Images/' + file_name)
    thispic = np.array(thispic, dtype=float)

    # Define some levels of reduction
    dMax = [25, 51, 170]
    (nr, nc, nd) = thispic.shape
    Reconstructed_image = np.zeros((nr, nc, nd), dtype=float)
    fig, axes = plt.subplots(1, len(dMax), figsize=(16, 4))
    # Go through each reduction level
    for d_idx, d in enumerate(dMax):
        # Go through each dimension of the image
        for i in np.arange(nd):
            data_i = thispic[:, :, i]
            # Model creation and run RBD
            RB = RBMs(data=data_i, tol=1e-6, col=d)
            bases, TransMat = RB.RBD()
            Reconstructed_image[:, :, i] = np.dot(bases, TransMat)

        # Display the reconstructed image
        axes[d_idx].imshow(Reconstructed_image.astype(np.uint8), cmap='viridis')
        axes[d_idx].set_title("Compressed to " + str(int(round(d/nr, 2)*100)) + "%")
        axes[d_idx].axis('off')  # Hide the axis
    # Save the figure
    plt.savefig('ReconstructedImage.png', dpi=300, bbox_inches='tight')
    plt.show()

