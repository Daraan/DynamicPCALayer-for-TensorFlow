# -*- coding: utf-8 -*-
"""
Created on Mon May 17 18:22:18 2021

@author: Daniel Sperber
Partialy based upon or code used from
https://github.com/scikit-learn
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from elbow_estimation import calculate_elbow

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
        purely pased on TensorFlow functions.
        Which is not 100% successfull yet and TensorFlow needs to run
        in eagerly mode.
        """
        # From Scikit
        # https://github.com/scikit-learn/scikit-learn/blob/15a949460dbf19e5e196b8ef48f9712b72a3b3c3/sklearn/utils/extmath.py#L504
        # Still not 100% TF
        max_abs_cols = tf.argmax(tf.abs(u), axis=0)
        print(max_abs_cols)
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
            Calcualte the SVD with this data
        center : bool, optional
            Center the data before applying SVD
            The default is True.
        flip_Vt : bool, optional
            Applies the svd_flip method.
            For a more deterministic svd result
            The default is True.

        Returns
        -------
        S : tf.Tensor
            SingularValues.
        U : tf.Tensor
            The second orthogonal matrix of the svd.
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
                
        if True: 
            # uses numpy, needs to run in eagerly mode
            if flip_Vt:
                npU, Vt = self.svd_flip(U.numpy(), Vt) # NOTE: NOT TF!
                U = tf.convert_to_tensor(npU)
                self.Vt = Vt
            else:
                self.Vt = Vt
        else: 
            # EXPERIMENTAL ues svt_flip_tf
            # Use TF only
            # Note: currently still uses .numpy() or non iterable max_abs_cols
            # above numpy version is faster.
            if flip_Vt:
                U, Vt = self.svd_flip_tf(U, Vt)
                self.Vt = Vt
            else:
                self.Vt = Vt
        
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
        # Todo use functools.cache
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
    
    use pcafit(data) method to caluclate the underlaying SVD
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
                'inverval' : every self.fit_interval batches 'fit_transform'
                              is used else 'transform'
                'fit' : caluclates the PCA but without transformation
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
        self._isfitted = _isfitted       # Used by partner layer, todo should not be private
        self.n_components = n_components
        self.partner_layer = partner_layer # NOTE: Set during init of partner 
        # For Experiments
        self.freeze_pca = freeze_pca     # interval_fit_transform performs only transform if True
        self.fit_interval = fit_interval
    
    def fitpca(self, X, n_components : int =None, set_mode : str = 'transform') -> None:
        """
        This is the method that usally is used to apply the necessary
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
        self.mode = mode
        if mode == 'fit_interval':
            self.fit_interval = fit_interval
    
    # call methods
    
    #@staticmethod
    def _call_pass(self, inputs):
        """
        Returns the input without doing anyhting.
        """
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
    # For possible serialization somewhen
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
        Estimates a good reduction by caluclating the loss at given points
        and applying the elbow method to the resulting graph.
        Increasing the Dimensions above the elbow value results only in a below
        average improvement.
        
        HINT:: 
        It is usefull to reduce the data to a certain size <1000 for this method to be efficient.
        The results of a small sample do not change relevantly.
        
        A custom test range can be passed, else it will cover the whole latent space.
        HINT: Passing only even dimensions will double the speed and this scales
        further.
        By halfing the test_range the results will only vary by +- 1
        

        TODO: A threshold to abort this test 
              if the values are not changeing relevantly.


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
            Currently no other mode is choosable.
            A faster method would be the calculation of the mean change
            and find the point where the absolute changes drop below this 
            value.
            But this method can fail for non convex/conkav graphs.
            
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
            Elbow Value and estiamted value for a good dimension
        List[float], optional
            if return_losses is set returns the losses in the test_range
        """
        BATCH_SIZE = 256 # Above 500 memory issues can occure!!
        losses = []
        test_range = test_range or range(encoded_data.shape[-1], 0, -1)
        reset_n, reset_mode = self.n_components, self._mode
        self.set_mode('transform')
        for i in test_range:
            if verbose:
                print(f"Reducing to {i}", end="\r")
            self.n_components=i;
            try: 
                # Calulates the loss over batches else can run out of memory
                # in this implementation
                partial_losses = []
                print("") # fix for spyder console.
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
                # todo disambigous if raise in custom loss, decoder function.
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


# %% Test Functions

def _sktransform(n, encoded):
    """
    For testing, maybe will be deprecated.
    A^1*A*latent - one transformation cycle
    """
    from sklearn.decomposition import PCA
    pca = PCA(n)
    latent = pca.fit_transform(encoded)
    return pca.inverse_transform(latent)


def use_loss_func(loss_func=bce_loss):
    """
    Decorator function that makes a native Keras / Tensorflow function
    compatible with the pca_tests function
    """    
    def calcloss(original, latent, decoder):
        return loss_func(original,  decoder.predict(latent)).numpy()
    return calcloss

def eval_loss(original, latent, decoder, loss_func=bce_loss):
    """
    Example of a native Keras / Tensorflow function to be used with
    pca_tests
    """
    return loss_func(original,  decoder.predict(latent)).numpy()



# NOTE: update python version for correct type hinting.
def pca_tests(original, encoded, 
                decoder : tf.keras.Model, 
                pca     : tf_PCA = None,
                include_last : bool = True, 
                start : int = 1, 
                min_rel_changes : Union[ float, List[float] ] = 0.0025, 
                test_funcs   : Dict[str, callable]   = {'pca_loss' : eval_loss},
                take_samples : Dict[str, List[int] ] = None,   # {Literal['indices'] : List[int]}=None, # python 3.8+
                min_samples  : int  = 20,
                verbatim     : bool = False ) -> Tuple[ Dict[str, List[int]], Dict[str, List[float]] ]:
    """
    Calculates the loss between the original and autoencoded data.
    For all n in [1, encoded.shape[-1]] the encoded data's dimension is
    reduced by PCA.
    Returns all used n and the losses x ,y
    
    The eval_func must accept three parameters
        eval_func(original, latent, decoder)
        This allows a individual calculation of the loss
        
        HINT:
        The decorator use_loss_func(loss_function) -> eval_func
        can be used here with standard functions.
        The loss_function should take two positional arguments:
        loss_function( original, decoded_result )
        
    Performs multiple test.
    Note that the returned sizes are for the longest test.
    You might need to zip them or do sizes[:len(results[i])]
    
    -- take_samples and min_samples
    These help to visualize results by taking samples of the decoded
    data for the different dimensions.
    min_samples assures that the tests by min_rel_changes are not terminated
    until this amount of samples has been taken.
    The samples are always taken at the provided indices, so for example
    pass one index for each possible class.
    
    It's possible to pass test_funcs = None to only take samples.
    
    Providing
    take_samples = {'indices': [int, ...]} 
    will take samples from the decoded data with the provided indices.
    These will be stored as 'index : [samples]' IN THE SAME dictionary


    Parameters
    ----------
    original : array like
        Original data
    encoded : array like
        Encoded data
    decoder : tf.keras.Model
        Model that has a .predict method for the encoded data.
    pca : tf_PCA, optional
        If pca object is provided it will NOT be fitted.
        Else a new one will be generated
        The default is None.
    include_last : bool, optional
        Perform the tests for no dimension result and append it. 
        NOTE: This does not influence take_samples, it will NOT take a sample
        of the decoded data.
        The default is True.
    start : int, optional
        Minimum dimension to start the test.
        The default is 1.
    min_rel_changes : Union[ float, List[float] ], optional
        A single value or a list of matching length with test_funcs.
        Will stop the individual tests if a relative change falls below
        this/these values.
        The default is 0.0025.
    test_funcs : Dict[str, callable], optional
        A dict with a names as keys and a callable to perform special tests.
        The callable has to take 3 parameters:
            eval_func(original, latent, decoder)
        Alternatively use the provided decorator
            use_loss_func(loss_function)
        and a loss_function( original, result )
        By default this calculates the binary_crossentropy
        The default is {'pca_loss' : use_loss_func(loss_func=bce_loss)}.
        
        
    take_samples : Dict["indices", List[int] ], optional
        Takes samples of the decoded data at the given indices
        for example for visualization.
        The samples will be inserted into this dictionary with the
        indices as new keys.
        The default is None.
    min_samples : int, optional
        Normally min_rel_changes end this test.
        If samples are to be taken this assures that this amount of samples
        are taken before the test is stoped.
        The default is 20.
    verbatim : bool, optional
        Dumps information of the current dimension and test results. 
        The default is False.

    Returns
    -------
    test_sizes : Dict[str, List[int]], 
        As keys the names and keys from test_funcs
        As values the dimensions where the tests where performed
    results : Dict[str, List[float]]
        As keys the names and keys from test_funcs
        As values the results of the functions in the test_funcs argument
    """
    # Check types
    if not isinstance(test_funcs, dict):
        raise TypeError("test_funcs must be a dict instance {name:function}. Can be empty {}.")
    if take_samples is not None and isinstance(take_samples, dict) \
        and not 'indices' in take_samples:
        raise KeyError("Provide a dict with a 'indices' key for take_samples.")
    
    if take_samples is not None:
        take_samples.update({idx:[] for idx in take_samples['indices'] if idx not in take_samples})
    samples_taken = 0
    
    if type(min_rel_changes) == float:
        min_rel_changes = [min_rel_changes] * len(test_funcs)
    if len(min_rel_changes) != len(test_funcs):
        raise ValueError("len(min_rel_changes) between 1 < changes < len(test_funcs)")
    
    # scikit or pcalayer
    if pca is None:
        pca = tf_PCA()
        pca.fit(encoded)

    results = {name:[] for name in test_funcs.keys()}

    sizes      = []
    test_done  = [False] * len(test_funcs)
    #start, end = [0.] * len(test_funcs), [0.] * len(test_funcs) # for times
    for n in range(start, encoded.shape[-1]):
        sizes.append(n)
        latent = pca.autoencode_transform(encoded, n) # reduce dimension and retransform
        
        for i, (test_name, eval_func) in enumerate(test_funcs.items()):
            if test_done[i]: 
                continue
            test_results = results[test_name]
            #start[i] = time.perf_counter()
            result = eval_func(original, latent, decoder)
            #end[i] = time.perf_counter()
            results[test_name].append(result)
            if len(test_results) > 5 and test_results[-2]*min_rel_changes[i] > np.abs(result - test_results[-2]):
                test_done[i] = True
                if verbatim:
                    print("Stopping Test for func " + test_name + str(i) + 
                          "\nchange was under min_rel_change. Continue with other tests.")
        # Store decoded samples by pca
        if take_samples is not None:
            samples = np.array([latent[s] for s in take_samples['indices']])
            decoded_samples = decoder.predict(samples)
            for sample, idx in zip(decoded_samples, take_samples['indices']):
                take_samples[idx].append(sample)
                    
        if verbatim:
            print("Results for size=", n, " : ", *[test_name+": "+str(results[test_name][-1])+ "," for test_name in test_funcs.keys()])
            #print("           Times:", [format(end[i]-start[i], ".2f")+"s" for i in range(len(test_funcs))])
        
        samples_taken+=1 # do not put in if
        # Note that all([]) is True if no test_funcs are given
        if all(test_done) and (take_samples is None or samples_taken >= min_samples):
            print("Stopping, changes were under min_rel_change. Only adding last value")
            break
    # Calculate for full model
    if include_last:
        latent = pca.autoencode_transform(encoded, encoded.shape[-1])
        for i, (test_name, eval_func) in enumerate(test_funcs.items()):
            results[test_name].append(eval_func(original, latent, decoder))
        sizes.append(encoded.shape[-1])
        # Todo: Also take sample here.
    # Get X values for the done tests
    test_sizes = {name : sizes[:len(results[name])] for name in results}
    return test_sizes, results # x, y




def loss_test(original, encoded, decoder, pca=None, calculate_last=True, min_rel_change=0.0025, eval_func=eval_loss,
              verbatim=False)-> Tuple[List[int], List[float]]:
    """
    This is an OUTDATED but simpler version of pca_tests that takes
    a single function.
    
    Calculates the loss between the original and autoencoded data.
    For all n in [1, encoded.shape[-1]] the encoded data's dimension is
    reduced by PCA.
    Returns all used n and the losses x ,y
    
    The eval_func must accept three parameters
    eval_func(original, latent, decoder)
    """
    if pca is None or not isinstance(pca, tf_PCA):
        print("Using Skitkit pca, much slower")
        scikit = True
    else:
        scikit = False
    
    results = []
    sizes = []
    for n in range(1, encoded.shape[-1]):
        sizes.append(n)
        if scikit:
            latent = _sktransform(n, encoded)
        else:
            pca.n_components_ = n
            latent = pca.transform(encoded) # decoder has inverse transform
        
        result = eval_func(original, latent, decoder)
        results.append(result)
        if verbatim:
            print("Result for", n, " : ", result)
        if len(results) > 10 and results[-2]*min_rel_change > np.abs(result - results[-2]):
            if verbatim:
                print("Stoping, loss change was under min_rel_change. Only adding last value")
            break
    if not scikit:
        pca.n_components_ = encoded.shape[-1]
        latent = pca.transform(encoded) # decoder has inverse transform 
    else:
        latent = encoded # lets assume perfect inverse tranfrom here
    results.append(eval_func(original, latent, decoder))
    sizes.append(encoded.shape[-1])
    return sizes, results # x, y
        

def plot_loss_reduce(losses, test_range, 
                     elbow_val=None, 
                     loss_func=bce_loss, 
                     data=None,
                     figtitle="", figheader="Loss change", variable_name="BCE Loss", show=True,
                     plot_minima=True, ylabel=None,
                     # parameters to combine figures
                     figsize=(9,6), newfig=True, **plotkwargs) -> Union[None, plt.Figure]:
    """
    Helps to visualize the results of pca_tests
    
    The loss_func argument is deprecated.
    
    data and data_size can be passed to calculate and plot an ideal loss on the orignal data.
    using data_size=None will calculate the loss for the complete data set but will take longer.
    """
    if newfig:
        fig = plt.figure(figsize=figsize)
    plt.title(figheader)
    plt.suptitle(figtitle)# + str(ENCODER_L1_VALUE))
    plt.xlabel("Latent Layer Size")
    plt.ylabel(ylabel or variable_name)
    plotkwargs.setdefault('label', variable_name)
    plotkwargs.setdefault('markersize',9)
    plotkwargs.setdefault('linewidth', 2)
    plt.plot(test_range, losses, ".-", **plotkwargs)
    if plot_minima:
        plt.axhline(min(losses), linestyle="--", c="C2", label="minima "+variable_name, linewidth=2.25)
    if elbow_val is not None:
        plt.axvline(elbow_val, linestyle=":", c="C4", label="Elbow value: "+str(elbow_val), linewidth=2.5)
    if data is not None and loss_func is not None:
        plt.axhline(loss_func(data, data), linestyle="--", c="C3", #label="ideal "+variable_name,
                    linewidth=2)
    plt.legend()
    if show:
        plt.show()
    if newfig:
        return fig


    