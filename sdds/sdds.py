
from .plot import plot_activations, plot_atoms, reconstruct
from .activations import ActivationMatrix
from .dictionary import Dictionary
from .csc import CSC
from .cdl import CDL

import numpy as np


class SDDS:

    def __init__(self, X, K, L, regularization, n_iter=100) -> None:

        # Original signal
        self.X = X.to_numpy()

        # Signal length
        self.N = len(X)

        # Number of atoms
        self.K = K

        # Length of the atoms
        self.L = L

        # Regularisation (lambda)
        self.regularization = regularization

        # Number of iterations
        self.n_iter = n_iter

        # Parametric dictionary
        self.D = self._generateDictionary()

        # Activation matrix
        self.Z = self._generateActivationMatrix()

    def process(self):
        """
        SDDS core.
        """

        csc = CSC(
            self.N, self.K, self.L,
            self.regularization,
            self.X, self.D, self.Z,
            method='ISTA'
        )
        cdl = CDL(self.N, self.K, self.L,
                  self.regularization,
                  self.X, self.D, self.Z)

        for t in range(self.n_iter):

            csc.step()
            cdl.step()
            self._log(t)

        return

    def _generateDictionary(self):
        """
        Generate a dictionary object.
        """
        return Dictionary(self.N, self.K, self.L)

    def _generateActivationMatrix(self):
        """
        Generate an activation vector object.
        """
        return ActivationMatrix(self.N, self.K, self.L)

    def getDictionary(self):
        """
        Helper function to get the dictionary.
        """
        return self.D.getDictionary()

    def getActivationMatrix(self):
        """
        Helper function to get the dictionary.
        """
        return self.Z.getActivations()

    def _log(self, t):
        """
        Print useful values after each iteration.
        """

        # Initialize verbose
        if t == 0:
            print('Iteration \t | \t Residuals \t ')
            print('------------------------------------')

        # Compute interesting values
        conv = np.sum([np.convolve(z_k, d_k)
                       for z_k, d_k
                       in zip(self.getActivationMatrix(),
                              self.getDictionary().T)], axis=0)
        residuals = np.linalg.norm(self.X - conv)**2
        print(f'{t} \t\t | \t {residuals:.4f}')

        return

    def plotAtoms(self, k):
        """
        Plot the first k atoms in the dictionary.
        """
        return plot_atoms(self.getDictionary(), k)

    def plotActivations(self):
        """
        Plot activations for all atoms.
        """
        return plot_activations(self)

    def reconstruct(self):
        """
        Reconstruct the signal using the atoms in the 
        dictionary.
        """
        return reconstruct(self.X, self.D, self.Z)
