import numpy as np
import scipy.optimize as optim
from sklearn.model_selection import train_test_split


class InterpolationRBF:
    """
    Class documentation.
    """

    def __init__(self):
        self.status = "Untrained"
        self.kernel_name = "No kernel"
        self.available_kernels = {
            "Gaussian": self.kernelGA,
            "C0 Matern": self.kernelM0,
            "C4 Matern": self.kernelM4,
            "Rational quadratic": self.kernelRQ
        }

    def kernelGA(self, x, sig):
        """
        Gaussian kernel (i.e., squared-exponential) :
            np.exp(-x**2/(2*hyperparam**2))
        """
        return np.exp(-x**2/(2*sig**2))

    def kernelM0(self, x, sig):
        """
        C^0 Matérn kernel: np.exp(-hyperparam*x)
        """
        return np.exp(-sig*x)

    def kernelM4(self, x, sig):
        """
        C^4 Matérn kernel:
            np.exp(-hyperparam*x)*(3+3*hyperparam*x+(hyperparam*x)**2)
        """
        return np.exp(-sig*x)*(3+3*sig*x+(sig*x)**2)

    def kernelRQ(self, x, sig):
        """
        Rational quadratic kernel with scale mixture parameter = 1.5:
            (1 + x**2/(2*2*hyperparameter**2))**-2
        """
        return (1 + x**2/(2*1.5*sig**2))**-1.5

    def train(self, samplex, sampley, hyperparam="optimize",
              kernel="Gaussian", valid_set_size=0.33):
        """
        Function to call to train the model.
        It creates the interpolation matrix for the given samples. Then the
        weight vector.
        You can both specify a float hyperparameter or ask for hyperparameter
        optimization by passing the string "optimization" for the argument
        hyperparameter. The optimization is based on a Leave One Out (LOO)
        error measure.
        """
        self.kernel_name = kernel
        self.kernel = self.available_kernels[kernel]

        if type(hyperparam) == int or type(hyperparam) == float:
            # If the hyperparameter is known
            self.hyperparameter = hyperparam
            weight_vec = self._weight_vector(
                samplex, sampley, self.hyperparameter)

        elif hyperparam == "optimize":
            # If the hyperparameter has to be optimized, first the training
            # dataset and the  validation dataset are separated
            tsetx, vsetx, tsety, vsety = train_test_split(
                samplex, sampley, test_size=valid_set_size)

            # Sample and dataset content is stored for later use (making prediction)
            self.samplex = samplex
            self.sampley = sampley
            self.train_samplex = tsetx
            self.train_sampley = tsety
            self.valid_samplex = vsetx
            self.valid_sampley = vsety

            # Optimize the hyperparameter. The initial value is the mean value of the
            # distance between sample coordinates
            sig_start = np.mean(np.linalg.norm(samplex, axis=1))
            res = optim.minimize(lambda x: self._optim_hyperparam(tsetx, tsety,
                                                                  vsetx, vsety,
                                                                  x),
                                 [sig_start], method='Nelder-Mead',
                                 bounds=((0.01, 100),), tol=1e-10)
            sigopt = res.x
            self.hyperparameter = sigopt

            if (abs(sigopt-0.01)<=0.0001) or (abs(sigopt-100)<=0.1):
                print("\n\tWARNING: The hyperparameter did not converge, the model might be overfitted.\n")

            # Compute the weight vector, needed for making predictions
            weight_vec = self._weight_vector(samplex, sampley, sigopt)

        else:
            raise ValueError(
                "The hyperparameter should either be a float or an int if you "
                "want a specific value, or the str 'optimize' if you want to "
                "let the program optimize the hyperparameter.")

        # Store the weight vector, the sample coordinates and indicates the
        # model is trained
        self.weight = weight_vec
        self.samplex = samplex
        self.status = "Trained"

    def predict(self, newpoints):
        """
        Once the model is trained, this function is used to predict the
        output for points that don't belong to the training/validation dataset.
        """
        if self.status == "Untrained":
            raise ValueError(
                "The model must be trained before making predictions.")
        else:
            # Compute the new interpolation matrix
            new_interpol_matrix = np.zeros(
                (newpoints.shape[0], self.samplex.shape[0]))

            for idx in range(newpoints.shape[0]):
                for jdx in range(self.samplex.shape[0]):
                    new_interpol_matrix[idx, jdx] = self.kernel(np.linalg.norm(newpoints[idx, :]
                                                                                - self.samplex[jdx, :]),
                                                                self.hyperparameter)

        return np.dot(new_interpol_matrix, self.weight)

    def _weight_vector(self, x, y, sig):
        """
        Function used for computing the weight vector for a given dataset and
        a given hyperparameter value.
        """
        # Compute the interpolation matrix which is (kernel(distance(point_i, point_j)))_ij
        interpol_matrix = np.zeros((x.shape[0], x.shape[0]))
        for idx in range(x.shape[0]):
            for jdx in range(x.shape[0]):
                interpol_matrix[idx, jdx] = self.kernel(
                    np.linalg.norm(x[jdx, :]-x[idx, :]), sig)

        # Compute and return the weight vector
        weight_vector = np.linalg.solve(interpol_matrix, y)

        return weight_vector

    def _optim_hyperparam(self, tsetx, tsety, vsetx, vsety, sig):
        """
        Cost function used to optimize the hyperparameter.
        """
        self.samplex = tsetx
        self.sampley = tsety

        self.weight = self._weight_vector(tsetx, tsety, sig)
        self.status = "Training"
        self.hyperparameter = sig

        cost_function = np.sum((vsety - self.predict(vsetx))**2)

        return cost_function
