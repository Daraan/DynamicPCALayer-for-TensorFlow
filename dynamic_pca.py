# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:22:18 2021

@author: Daniel Sperber
Partialy based upon or code used from
https://github.com/scikit-learn
"""

import tensorflow as tf
import numpy as np

from utils.elbow_calculation import calculate_elbow

from typing import List, Dict, Tuple, Union, Callable, Iterable # Literal
import functools

bce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0.0, reduction="auto", name="binary_crossentropy"
       )

# =============================================================================
# %% SVD and PCA (nearly) in Tensorflow
# =============================================================================


class tf_SVD():
    @staticmethod    
    def svd_flip_tf(u, v) -> Tuple[tf.Tensor]:
        """
        See svd_flip
        
        This tries to implement it a deterministic output from SVD
        purely based on TensorFlow functions.
        Which is not 100% successful yet and TensorFlow needs to run
        in eagerly mode.
        """
        # From Scikit
        # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/utils/extmath.py#L504
        # Still not 100% TF
        max_abs_cols = tf.argmax(tf.abs(u), axis=0)
        #print(max_abs_cols)
        # TF does not support fancy np slicing
        signs = tf.sign(u.numpy()[max_abs_cols, range(u.shape[1])]) # numpy -> Needs to run eagerly / dynamic

        
        # Much slower as the one above, max_abs_cols does not support iteration, todo find other method
        #signs = tf.gather_nd(u, indices=(tuple(zip(max_abs_cols, range(u.shape[1])))))

        u *= signs
        v *= signs[:, tf.newaxis] # make compatible and change signs
        return u, v
    
    @staticmethod
    def svd_flip(u, v) -> Tuple[tf.Tensor]:
        """
        Taken from Scikit
        https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/utils/extmath.py#L504
        NOTE: Not pure TensorFlow but faster compared to svd_flip_tf
        
        Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
        v : ndarray
            u and v are the output of `linalg.svd` or
            :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
            dimensions so one can compute `np.dot(u * s, v)`.
            The input v should really be called vt to be consistent with scipy's
            ouput.
        """
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
        return u, v
    
    def __init__(self, X=None, center : bool =True, flip_Vt : bool =True) -> None:
        if X is not None:
            self._fit(X, center, flip_Vt)

    
    def _fit(self, X, center : bool=True, flip_Vt : bool=True) -> Tuple[tf.Tensor]:
        """
        Calculates the SVD of a matrix X

        Parameters
        ----------
        X : tf.Tensor or array like
            Calculate the SVD with this data
        center : bool, optional
            Center the data before applying SVD
            The default is True.
        flip_Vt : bool, optional
            Applies the svd_flip method.
            For a more deterministic SVD result
            The default is True.

        Returns
        -------
        S : tf.Tensor
            SingularValues.
        U : tf.Tensor
            The second orthogonal matrix of the SVD.
        V : tf.Tensor
            Transformation Matrix.

        """
        if len(X.shape) == 2:
            self.n_samples, self.n_features = X.shape
        else:
            #self.n_samples, self.n_features, y, channels = X.shape
            self.n_samples = X.shape[0]
        
        self.mean = tf.reduce_mean(X, axis=0) 
        
        if center:
            X = tf.subtract(X, self.mean)
        
        S, U, V = tf.linalg.svd(X, full_matrices=False)
        # NOTE: U and thereby svd_flip might not the same as scipy.linalg.svd 
        # and maybe np.linalg.svd
        self.S, self.U = S, U
        
        Vt = tf.transpose(V)
                
        #if True: 
        # uses numpy, needs to run in eagerly mode
        if flip_Vt:
            npU, Vt = self.svd_flip(U.numpy(), Vt) # NOTE: NOT TF!
            U = tf.convert_to_tensor(npU)
            self.Vt = Vt
        else:
            self.Vt = Vt
        #else: 
            # EXPERIMENTAL use svt_flip_tf
            # Use TF only
            # Note: currently still uses .numpy() or non iterable max_abs_cols
            # above numpy version is faster.
        #    if flip_Vt:
        #        U, Vt = self.svd_flip_tf(U, Vt)
        #        self.Vt = Vt
        #    else:
        #        self.Vt = Vt
        
        self.S, self.U, self.V = S, U, V
        return S, U, V
    
    @functools.wraps(_fit)
    def fit(self, X, center=True, flip_Vt=True):
        # Can and will be shadowed in the PCA layer
        return self._fit(X, center, flip_Vt)
    

class tf_PCA(tf_SVD):  
    def __init__(self, X=None, n_components : int =None):
        # fit during init
        self._isfitted = False
        self.n_components=n_components
        self.mean = "Not fitted yet"
        super().__init__(X, center=True)

    def fit(self, X, n_components=None):
        """
        Calculates the SVD of the given data X.
        Optionally the attribute n_components to choose the resulting
        dimension of the transformation.

        Parameters
        ----------
        X : tf.Tensor or array like
            Data.
        n_components : int, optional
            Resulting dimension of the transformation 
            The default is None.
        
        Set Attributes
        ----------
        This sets 
            self.n_samples = X.shape[0]
            self.n_components = n_components or self.n_components
            self.mean ; of the data
            self.U, self.S, self.V
            self.S are the singular values

        Returns
        -------
        self : tf_PCA
            the object itself

        """
        self.n_samples = X.shape[0]
        self.n_components = n_components or self.n_components
        #self.mean = tf.reduce_mean(X, axis=0) # set in parent
        super().fit(X, center=True, flip_Vt=True)
        self._isfitted = True
        return self
    
    def transform(self, X, n_components=None):
        """
        Transforms the data with the PCA and reduces the dimension
        if n_components or self.n_components is set.
        """
        n_components = n_components or self.n_components
        X = tf.subtract(X, self.mean)    # Center data
        X_transformed = tf.matmul(X, tf.transpose(self.Vt[:n_components, :])) # todo check necessity of transpose
        return X_transformed
        
    def fit_transform(self, X, n_components=None):
        """
        Applies the fit and transform method
        """
        self.fit(X, n_components)
        return self.transform(X, n_components)
        
    def inverse_transform(self, X_transformed):
        """
        NOTE and deprecation warning: will be replaced by invert_transform
        
        Reverses the transformation back to the original space and dimension
        """
        X_hat = tf.matmul(X_transformed, self.Vt[:X_transformed.shape[-1], :]) # Always transforms back to full
        return X_hat + self.mean        # shift back
    
    invert_transform = inverse_transform
    
    def autoencode_transform(self, X, n_components=None):
        """
        Performs transform and invert_transform.
        This can be used to test the information loss of the PCA
        """
        return self.invert_transform(self.transform(X, n_components))
    
    @property
    def n_components_(self):
        # to be scikit compatible
        return self.n_components
    
    @n_components_.setter
    def n_components_(self, value : int):
        if not isinstance(value, int):
            raise TypeError("Not an int")
        self.n_components = value
    
    # #############################################
    #### Additional features, non purely TensorFlow
    # #############################################
    
    @property
    def singular_values_(self) -> np.array:
        if not self._isfitted:
            raise ValueError("Object is not fitted yet.")
        return self.S.numpy()
    
    @property
    def variance(self) -> np.array:
        return self.get_variance(self.n_components)
    
    def get_variance(self, n_components=None) -> np.array:
        if not self._isfitted:
            raise ValueError("Object is not fitted yet.")
        # create on demand
        # TODO use functools.cache
        if "explained_variance_" not in self.__dict__:
            self.explained_variance_ = np.array((self.S ** 2) / (self.n_samples - 1)) # -1 for bias
        return self.explained_variance_[:n_components]
    
    @property
    def variance_ratio(self):
        # todo could do this cheaper via singular values ratio
        return self.get_variance_ratio(self.n_components)
    
    def get_variance_ratio(self, n_components=None, cummulated=True):
        if "explained_variance_ratio_" not in self.__dict__:
            explained_variance_ = self.get_variance(n_components=None)
            total_var = explained_variance_.sum()
            self.explained_variance_ratio_ = explained_variance_ / total_var
        
        if cummulated:
            return np.cumsum(self.explained_variance_ratio_[:n_components])
        return self.explained_variance_ratio_[:n_components]
    
    def get_noise_variance(self, n_components):
        # From Scikit
        # https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/decomposition/_pca.py#L489
        # "Equal to the average of (min(n_features, n_samples) - n_components) 
        # smallest eigenvalues of the covariance matrix of X"
        # maximum likelihood for the rest variance.
        if n_components is None or n_components >= len(self.variance):
            return 0.0 # NOTE: Not a tensor
        
        noise_var = tf.reduce_mean(self.variance[n_components:], axis=0)
        return noise_var
    
    
    def scipy_get_covariance(self) -> np.array:
        """
        Compute data covariance with the generative model.
        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        components_ = self.components_
        exp_var = self.explained_variance_
        if self.whiten:
            components_ = components_ * np.sqrt(exp_var[:, np.newaxis])
        exp_var_diff = np.maximum(exp_var - self.noise_variance_, 0.)
        cov = np.dot(components_.T * exp_var_diff, components_)
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify diag inplace
        return cov
    
    def get_covariance_old(self, n_components=None, whiten=False) -> tf.Tensor:
        """
        # Source: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/decomposition/_base.py#L25
        Compute data covariance with the generative model.
        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        Returns
        -------
        cov : array, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        exp_var_full = self.get_variance()
        if n_components is None or n_components >= self.S.shape[0]:
            n_components = self.S.shape[0]
            var_noise = 0.
        else:
            var_noise = tf.reduce_mean(exp_var_full[n_components:], axis=0)
        
        components_ = self.Vt[:n_components]
        exp_var = exp_var_full[:n_components] # rest is the noise
        if whiten:
            components_ = components_ * tf.sqrt(exp_var[:, tf.newaxis])
        exp_var_diff = tf.maximum(exp_var - var_noise, 0.) # element wise maximum
        cov = tf.matmul(tf.transpose(components_) * exp_var_diff, components_).numpy()
        
        cov.flat[::len(cov) + 1] += self.noise_variance_  # modify the diag inplace
        return cov
    
    
    def get_covariance(self, n_components=None, whiten=False) -> tf.Tensor:
        """
        # 
        Based on Source: https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/decomposition/_base.py#L25
        
        Compute data covariance with the generative model.
        ``cov = components_.T * S**2 * components_ + sigma2 * eye(n_features)``
        where S**2 contains the explained variances, and sigma2 contains the
        noise variances.
        Returns
        -------
        cov : tf.Tensor, shape=(n_features, n_features)
            Estimated covariance of data.
        """
        var_full = self.get_variance()
        if n_components is None or n_components >= self.S.shape[0]:
            # So we can work with split
            n_components = self.S.shape[0]
            components_ = self.Vt
            var = var_full
            noise = 0.
            cov = tf.matmul(tf.transpose(components_) * var, components_)
        else:
           components_ = self.Vt[:n_components]
           var, noise = tf.split(self.get_variance(), (n_components, -1))
           noise = tf.reduce_mean(noise, axis=0)
           print("Noise", noise)
           
           exp_var_diff = tf.maximum(var - noise, 0.) # element wise maximum, todo numpy
           print(exp_var_diff)
           cov = tf.matmul(tf.multiply(tf.transpose(components_), exp_var_diff), components_)
           
           print([noise.numpy()]*self.S.shape[0])
           cov += tf.linalg.diag([noise.numpy()]*self.S.shape[0])
        
        # todo fix this:
        if whiten:
            components_ = components_ * tf.sqrt(var[:, tf.newaxis])
        
        return cov
    
    def get_correlation(self, n_components : int = None) -> np.array:
        """
        Get the correlation matrix R of the data. 
        Computed with the covariance matrix C
        R_ij = C_ij / sqrt(C_ii * C_jj)
        """
        # possible to call this via fit cor = np.corrcoef(data.T)
        cov = self.get_covariance(n_components) # Get C
        d = np.sqrt(np.diag(cov))   # sqrt(C_ii)
        outer_v = np.outer(d, d)    # sqrt(C_ii) * sqrt(C_jj) as n x n matrix, 
        cor = cov / outer_v         # element wise division. todo: unlikely division by 0.
        cor[cov == 0] = 0            # In case of divion by 0 and nan values.
        return cor
    
    def _test(self, X, n_components=None):
        """
        This is a test function to see if it works.
        """
        # USE PRE ENCODED DATA!
        self.fit(X, n_components)
        print("Input:", X[0])
        XT = self.transform(X, n_components)
        #U = self.U[:, :n_components]
        #U *= self.S[:n_components]
        #X = tf.subtract(X, self.mean)
        #XT = tf.matmul(X, tf.transpose(self.Vt[:n_components, :]))
        print("Transformed", XT[0].numpy(), "\nXTransformed and U shape\-n", XT.numpy().shape, self.U.shape)
        print("Output", self.inverse_transform(XT).numpy()[0])
    
    # Save & Load
    
    def get_config(self):
        """
        TODO not yet compatible with Tensorflow!
        """
        config = {"S":self.pca.S, "U":self.pca.U, "V":self.pca.V, "Vt":self.pca.Vt,
                  "mean" : self.mean,
                  "n_samples" : self.pca.n_samples, "n_components" : self.pca.n_components}
        # maybe not config but state.
        return config    
    
    # Alternatively reconstruct via encoded data
    @classmethod
    def from_config(cls, config):
        """
        TODO: NOT FINAL!
        """
        pca = cls()
        # maybe just do update __dict__
        pca.S = config["S"]
        pca.U = config["U"]
        pca.V = config["V"]
        pca.Vt = config["Vt"]
        pca.n_samples = config["n_samples"]
        pca.n_components = config["n_components"]
        pca.mean = config['mean']
        pca._isfitted = True # well else wouldn't make sense
        return pca
            


class PCALayer(tf.keras.layers.Layer):
    """
    Dynamic PCA Layer
    
    use pcafit(data) method to calculate the underlying SVD
    and the n_components attribute to apply the PCA and dimension
    reduction via this layer.
    This can be done at any point.
    """
    
    n_components : int      # Which dimension the results will be
    weights=[] # TODO: store svd as weights
    layers =[] # Old and possible idea to nest two layers into one. todo: can be removed
    
    def __init__(self, mode='pass', 
                 n_components : int = None,
                 name : str = "PCATransform",
                 *, partner_layer : "PCAInverseLayer" = None, 
                 fit_interval : int =100,
                 pre_encoder : tf.keras.Model = None,
                 data : list = None, 
                 freeze_pca : bool =False, # to be used by loading methods
                 _isfitted=False, # to be used by loading methods
                 **kwargs):
        """
        Parameters
        ----------
        mode : str, optional
            Mode of the PCA layer how the call and transformation is applied.
            The default is 'pass', after fitpca is called the mode changes
            to 'transform'.
            Valid modes are:
                'pass'      : Does nothing and passes input
                'transform' : applies PCA transformation
                'fit_transform' : calculates and applies the PCA
                'interval' : every self.fit_interval batches 'fit_transform'
                              is used else 'transform'
                'fit' : calculates the PCA but without transformation
        fit_interval : int, optional
            For the 'interval' mode. Defines are how many batches the pca
            should be fitted again.
            The default is 100.
        
        freeze_pca : bool, optional
            Mainly to be used with loading methods.
            The freeze_pca attribute can be set to make a fit_transform
            mode similar to 'transform'.
            Possible to be deprecated!
            The default is False.
        
        # Experimental and likely removed in the future
        # These can be used by a non_batch version of fit_transform
        # via a callback for example
        pre_encoder: Model with all layers before this one.
        data : The training data
            With pre_encoder and data the PCA can be performed on more samples
            than contained by the batch. 
            WARNING: Needs more of memory and might fail.
        
        **kwargs : optional arguments for the keras.layers.Layer class

        Returns
        -------
        None.

        """
        kwargs['dynamic']   = True     # With current svd_flip, needs to run eagerly
        kwargs['trainable'] = False
        kwargs['name'] = name
        #kwargs['weights'] = []
        super().__init__(**kwargs)
        self.set_mode(mode, fit_interval)
        self.pca = tf_PCA()
        self.counter = 0
        self.data = data
        self._isfitted = _isfitted       # Used by partner layer, todo should not be private (really?)
        self.n_components = n_components
        self.partner_layer = partner_layer # NOTE: Set during init of partner 
        # For Experiments
        self.freeze_pca = freeze_pca     # interval_fit_transform performs only transform if True
        self.fit_interval = fit_interval
    
    def fitpca(self, X, n_components : int =None, set_mode : str = 'transform') -> None:
        """
        This is the method that usually is used to apply the necessary
        actions to perform the PCA

        Parameters
        ----------
        X : 
            The data to be fitted
        n_components : int, optional
            Set a value for the resulting dimension of the PCA.
            This value can be changed via the n_components attribute
            The default is None.
        set_mode : str, optional
            If set uses the string to set the mode.
            The default is 'transform'.
        """
        if set_mode:
            self.set_mode('transform') # raise first in case it's invalid
        self.n_components = n_components or self.n_components
        self.pca.fit(X, self.n_components)
        self._isfitted = True
    
    @property
    def mean(self):
        return self.pca.mean
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, newmode):
        self._mode = newmode
        if newmode == 'transform':
            self._call = self._call_transform
        elif newmode == 'fit_transform':
            self._call = self._call_fit_transform
        elif newmode == 'fit':
            self._call = self._call_fit
        elif newmode == 'interval':
            self._call = self._call_interval_fit_transform
        elif newmode == 'pass':
            self._call = self._call_pass
        else:
            raise ValueError(newmode + " is not a valid newmode")
            
    def set_mode(self, mode:str, fit_interval=100):
        """
        Changes the behavior how the layer behaves when called.
        
        ------
        
        mode : str, optional
            Mode of the PCA layer how the call and transformation is applied.
            The default is 'pass', after fitpca is called the mode changes
            to 'transform'.
            Valid modes are:
                'pass'      : Does nothing and passes input
                'transform' : applies PCA transformation
                'fit_transform' : calculates and applies the PCA
                'interval' : every self.fit_interval batches 'fit_transform'
                              is used else 'transform'
                'fit' : calculates the PCA but without transformation
        fit_interval : int, optional
            For the 'interval' mode. Defines are how many batches the pca
            should be fitted again.
            The default is 100.
        """
        self.mode = mode
        if mode == 'fit_interval':
            self.fit_interval = fit_interval
    
    # call methods
    
    #@staticmethod
    def _call_pass(self, inputs):
        """Returns the input without doing anything."""
        return inputs # maybe just shift by mean
    
    def _call_transform(self, inputs):
        # Assert is fitted
        return self.pca.transform(inputs, self.n_components)
    
    def _call_fit_transform(self, inputs, batch_only=True, pca_samples=1000):
        if batch_only:
            transformed = self.pca.fit_transform(inputs)
            return transformed
        # Possible to apply pca over all data, but NOTE:
        self.pca.fit(self.pre_encoder(self.data[:pca_samples])) # NEEDS A LOT OF MEMORY. Might fail.
        self._isfitted = True 
        return self.pca.transform(inputs)
    
    def _call_interval_fit_transform(self, inputs):
        """
        Performs the PCA after a set amount of batches defined by self.fit_interval.
        All further batches are transformed transformed with the pca from that batch.
        """
        #print(inputs.shape, self.counter)
        if not self.freeze_pca and not (self.counter % self.fit_interval):
            transformed = self._call_fit_transform(inputs)
        else:
            transformed = self._call_transform(inputs)
        
        self.counter += 1
        return transformed
      
    def _call_fit(self, inputs):
        """
        Fits with inputs and returns unchanged inputs.
        Inverse PCA layer with perform retransform on the changed values.
        """
        # Not sure how much sense this one makes but anyway.
        self.pca.fit(inputs)
        self._isfitted = True 
        return inputs
    
    def _call(self, inputs):
        # set_mode replaces this one.
        raise AttributeError("_call mode has not been specified.")
    
    @functools.wraps(tf.keras.layers.Layer.call)
    def call(self, inputs):
        """
        This will apply the PCA or just pass the inputs depending
        on the set mode.
        """
        return self._call(inputs) # gets changed by set_mode
    
    @functools.wraps(tf.keras.layers.Layer.compute_output_shape)
    def compute_output_shape(self, input_shape):
        # In tf.keras not always used!
        output_shape = tf.TensorShape([input_shape[0], self.n_components or input_shape[1:]])
        return output_shape
        

     
    # TODO
    # For possible serialization in the future
    # Need to store the SVD matrices
    @functools.wraps(tf.keras.layers.Layer.get_config)
    def get_config(self) -> dict:
        """
        TODO: Improve and remove matrices.
        """
        config = super().get_config()
        config.update({"mode": self._mode, "fit_interval": self.fit_interval,
                       "freeze_pca" : self.freeze_pca, "_isfitted" : self._isfitted,})
        # Todo & howto, save pca
        
        # additional
        pca_config = {"S":self.pca.S, "U":self.pca.U, "V":self.pca.V, "Vt":self.pca.Vt,
                      'mean' : self.mean,
                      "n_samples" : self.pca.n_samples, "n_components" : self.pca.n_components}
        config['pca_config'] = pca_config
        return config
   
    # Todo
    @functools.wraps(tf.keras.layers.Layer.from_config)
    @classmethod
    def from_config(cls, config):
        """
        TODO: Improve.
        Creates layer from a given config
        """
        config = config.copy()
        pca = tf_PCA.from_config(config.pop('pca_config')) # NOTE: update if name changes!
        layer = cls(**config)
        layer.pca = pca
        return layer
    
    @classmethod
    def from_encoder(cls, encoder, data, config):
        """
        TODO: Experimental
        
        Working load function, uses an encoder to generate a new pca.
        """
        encoded_data = encoder.predict(data)
        layer = cls(**config)
        layer.fitpca(encoded_data, )
        return layer
    
    #
    def estimate_reduction(self, data   : tf.Tensor, 
                           encoded_data : tf.Tensor,
                           decoder : tf.keras.Model, 
                           *, method : str='elbow', 
                           loss_func : Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = bce_loss,
                           return_losses : bool =True, 
                           test_range    : Iterable =None, 
                           adjust_components : bool = False,
                           verbose : bool =True) -> Tuple[int, List[float]]:

        """
        Estimates a good reduction by calculating the loss at given points
        and applying the elbow method to the resulting graph.
        Increasing the Dimensions above the elbow value results only in a below
        average improvement.
        
        HINT:
        It is useful to use only a data sample <1000 for this method to be efficient.
        The results of a small sample do not change relevantly  - but needs to be tested for the dataset.
        
        A custom test range can be passed, else it will cover the whole latent space.
        HINT: Passing only even dimensions will double the speed and this scales
        further.
        By halving the test_range the results will only vary by +- 1
        Likewise the speed can be increased linearly with the deviation.

        TODO: A threshold to abort this test 
              if the values are not changing relevantly.


        Parameters
        ----------
        data : tf.Tensor
            DESCRIPTION.
        encoded_data : tf.Tensor
            encoded_data the pre_encoded data WITHOUT PCA applied.
        decoder : tf.keras.Model
            The decoder including PCAInverse layer if present
            
        Keyword arguments
        ----------
        
        method : str, optional
            Method to estimate the results
            The default is 'elbow'.
            NOTE: Currently no other mode can be chosen.
            A faster method would be the calculation of the mean change
            and find the point where the absolute changes drop below this 
            value.
            But this method can fail for non convex/concave graphs.
            
        loss_func : Callable[tf.Tensor, tf.Tensor], optional
            Function for the loss calculation
            The default is bce_loss.
        return_losses : bool, optional
            DESCRIPTION. 
            The default is True.
        test_range : Iterable, optional
            See above. The dimensions that shall be tested.
            NOTE: Very low dimensions are necessary, very high ones less.
                If there is nearly no improvement for high dimensions
                excluding them will not only improve the speed but also
                yield a better result.
            The default is None.
        adjust_components : bool, optional
            If True sets the n_components attribute to the resulting value
            The default is False.
        verbose : bool, optional
            Prints which dimension is currently tested
            The default is True.


        Returns
        -------
        int
            Elbow Value and estimated value for a good dimension
        List[float], optional
            if return_losses is set returns the losses in the test_range
        """
        BATCH_SIZE = 256 # Above 500 memory issues can occur! TODO: can these be handled here?
        losses = []
        test_range = test_range or range(encoded_data.shape[-1], 0, -1)
        reset_n, reset_mode = self.n_components, self._mode
        self.set_mode('transform')
        for i in test_range:
            if verbose:
                print(f"Reducing to {i}", end="\r")
            self.n_components=i;
            try: 
                # Calculates the loss over batches else can run out of memory
                # in this implementation
                partial_losses = []
                print("") # fix for Spyder console. # todo check for newer version
                for k in range(len(encoded_data) // BATCH_SIZE + 1):
                    out = decoder(self.pca.transform(encoded_data[k * BATCH_SIZE : (k+1) * BATCH_SIZE], i))
                    ploss = loss_func(data[k*BATCH_SIZE : (k+1) * BATCH_SIZE], out)
                    if isinstance(ploss, tf.Tensor): # allows for other non tf functions
                        ploss = ploss.numpy()
                    partial_losses.append(ploss)
                loss = np.mean(partial_losses)
            except tf.errors.InvalidArgumentError as e:
                # current VAE setup with custom loss needs n to be even.
                # copying seams better than inserting None, nan   
                # todo ambiguous if raise in custom loss, decoder function.
                loss = losses[-1]
                print(e)
            except Exception as e:  
                print(e)        
                raise
            losses.append(loss)
            # todo add abort if change is close to 0    
                
            #if False and not (i % 4) and i<50:
            #    plot_digit(decoder.predict(self(encoded_data[idx:idx+1])), show=False)
            #    plt.title("PCA reduction: "+str(i) +" / "+str(encoded_data.shape[-1]))
            #    plt.show()

        elbow_val = calculate_elbow(losses, test_range)
        print("Elbow and suggested reduction at", elbow_val)
        self.n_components = elbow_val if adjust_components else reset_n
        self._mode = reset_mode
        self.pcalosses = losses
        self.elbow_val = elbow_val
        if return_losses:
            return elbow_val, losses
        return elbow_val
        

class PCAInverseLayer(tf.keras.layers.Layer):
    """
    The partner layer of a PCALayer to reverse the transformation.
    And is simply constructed via a PCALayer
    pca_layer = PCALayer()
    pca_reverse_layer = PCAInverseLayer(pca_layer)
    """
    
    weights=[] # todo use appropriately
    layers =[] # todo remove
    
    def __init__(self, partner_layer : PCALayer = None, name="PCAInverseTransform", **kwargs):
        """
        

        Parameters
        ----------
        partner_layer : PCALayer, optional
            DESCRIPTION. The default is None.
        name : TYPE, optional
            DESCRIPTION. The default is "PCAInverseTransform".
        **kwargs : 
            Keyword arguments for tf.keras.layers.

        Returns
        -------
        None.

        """
        kwargs['dynamic']   = True  # Actually is not dynamic. Todo test.
        kwargs['trainable'] = False
        kwargs['name'] = name
        #kwargs['weights'] = []
        super().__init__(**kwargs)
        if partner_layer is not None:
            self.partner_layer = partner_layer
            partner_layer.partner_layer = self   
            self.pca = partner_layer.pca
        else:
            print("Layer created without a partner. Be sure to set layer.partner_layer manually.")
    
    @property
    def partner_layer(self):
        return self._partner_layer
    
    @partner_layer.setter
    def partner_layer(self, layer):
        self._partner_layer = layer
        self.pca = layer.pca
    
    def compute_output_shape(self, input_shape):
        # In tf.keras not always used!
        print("in shape", input_shape)
        if not self.pca._isfitted or self.partner_layer.mode == 'pass':
            print("returning input")
            return input_shape
        print("Is fitted returning shape:", tf.TensorShape([None, self.pca.S.shape[0]]))
        return tf.TensorShape([None, self.pca.S.shape[0]])

    @functools.wraps(tf.keras.layers.Layer.call)
    def call(self, inputs):
        if not self.partner_layer.pca._isfitted or self.partner_layer.mode == 'pass':
            return inputs # Note might be problematic with dimensions?
        return self.pca.inverse_transform(inputs, n_components=self.partner_layer.n_components)

    @functools.wraps(tf.keras.layers.Layer.get_config)
    def get_config(self):
        config = super().get_config()
        # NOTE: Does not work for saving!
        config['partner_layer'] = self.partner_layer
        return config

