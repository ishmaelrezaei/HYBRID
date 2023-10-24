import numpy as np
class RBMs:
    def __init__(self, data, tol, col):
        self.data = data
        self.tol = tol
        self.col = col

    def RBD(self):
        data = self.data
        tol = self.tol
        col = self.col

        data = data.astype(float)
        # Praparation of the algorithm
        (nr, nc) = data.shape;
        bases = np.zeros((nr, col), dtype=float);
        TransMat = np.zeros((col, nc), dtype=float);
        AtAAtXi = np.zeros((nc, col), dtype=float);
        xiFlag = np.zeros(col, dtype=int);
        xiFlag[0] = np.random.randint(nc);
        xiFlag[0] = 0
        i = 0;
        CurErr = tol + 1;

        # Preparation for efficient error evaluation
        ftf = np.sum(data.T * data.T, axis=1);

        # The RBD greedy algorithm
        while (i < col) and (CurErr > tol):
            biCand = data[:, xiFlag[i]];
            # Inside: Gram-Schmidt orthonormalization of the current candidate with all previsouly chosen basis vectors
            for j in np.arange(i):
                biCand = biCand - np.dot(biCand.T, bases[:, j]) * bases[:, j];
            normi = np.linalg.norm(biCand);

            if normi < 1e-7:
                print("Reduced system getting singular - to stop with", i - 1, "basis functions")
                bases = bases[:, :i - 1];
                TransMat = TransMat[:i - 1, :];
                break
            else:
                bases[:, i] = (biCand / normi).flatten();

            TransMat[i, :] = np.dot(bases[:, i].T, data);

            # Inside: With one more basis added, we need to update what allows for the efficient error evaluation.
            AtAAtXi[:, i] = np.dot(data.T, bases[:, i]);
            # Inside: Efficiently go through all the columns to identify where the error would be the largest if we were to use the current space for compression.

            TMM = TransMat[:i+1, :];
            te1 = np.sum(AtAAtXi[:, :i+1] * TMM.T, axis=1);
            te2 = np.sum(TMM.T * TMM.T, axis=1);
            errord = ftf - 2 * te1 + te2;
            CurErr = np.max(errord);
            TempPos = np.argmax(errord);

            # Mark this location for the next round
            if (i < col - 1):
                xiFlag[i + 1] = TempPos;
            # If the largest error is small enough, we announce and stop.
            if CurErr <= tol:
                print(CurErr)
                print("Reduced system getting singular - to stop with", i - 1, "basis functions")
                bases = bases[:, :i + 1];
                TransMat = TransMat[:i + 1, :];
            else:
                i += 1

        return bases, TransMat
