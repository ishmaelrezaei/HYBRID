# [HYper-reduced Basis Reduction via Interactive Decomposition](https://ishmaelrezaei.github.io/HYBRID/) in Python

In this project, we developed a high-speed dimensional reduction algorithm designed for efficiently handling massive matrices. Our work involved enhancing the Reduced Basis Dimension (RBD) algorithm by incorporating numerical solutions for partial differential equations. We are currently preparing to release the code for the HYBRID algorithm following its paper publication.

In the meantime, we offer access to the RBD algorithm, complete with two practical examples: image compression and video compression. It's important to note that the primary goal isn't limited to the application of HYBRID or RBD in video/image compression; instead, it focuses on using images and videos to effectively illustrate the efficiency of these algorithms.

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

To see an example, please run the file `one_dimensional_example.py` in the folder `/PGPs_autograd/Examples/` or `/PGPs_tensorflow/Examples/`.

    # let X, y be the loaded data
    # Model creation:
    pgp = PGP(X, y, M = 8, max_iter = 6000, N_batch = 1)
    
    # Training
    pgp.train()
    
    # Prediction
    mean_star, var_star = pgp.predict(X_star)

## Installing Dependencies

This code depends on `autograd` (https://github.com/HIPS/autograd), `tensorflow` (https://www.tensorflow.org/), `numpy` (http://www.numpy.org/), `scikit-learn` (http://scikit-learn.org/stable/index.html), `matplotlib` (https://matplotlib.org/), `pyDOE` (https://pythonhosted.org/pyDOE/), and `pandas` (http://pandas.pydata.org/) which can be installed using

    pip install matplotlib
    pip install cv2
    pip install numpy
    pip install PIL
    pip install sys
    
