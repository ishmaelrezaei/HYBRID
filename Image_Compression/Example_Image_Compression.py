# Image compression
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
sys.path.insert(0, '../RBMs_Functions/')
from RBMs_Functions import RBMs
sys.path.pop(0)

if __name__ == "__main__":
    # Image compression
    file_name = 'Lena.png';
    thispic = Image.open('Images/' + file_name);
    thispic = np.array(thispic, dtype=float)

    dMax = [25, 51, 170];
    (nr, nc, nd) = thispic.shape;
    Reconstructed_image = np.zeros((nr, nc, nd), dtype=float);
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for d_idx, d in enumerate(dMax):
        for i in np.arange(nd):
            data_i = thispic[:, :, i];
            RB = RBMs(data=data_i, tol=1e-6, col=d)
            bases, TransMat = RB.RBD();
            Reconstructed_image[:, :, i] = np.dot(bases, TransMat);

        # Display the reconstructed image
        axes[d_idx].imshow(Reconstructed_image.astype(np.uint8), cmap='viridis')
        axes[d_idx].set_title("Compressed to " + str(int(round(d/nr, 2)*100)) + "%")
        axes[d_idx].axis('off')  # Hide the axis
    # Save the figure
    plt.savefig('ReconstructedImage.png', dpi=300, bbox_inches='tight')
    plt.show()

