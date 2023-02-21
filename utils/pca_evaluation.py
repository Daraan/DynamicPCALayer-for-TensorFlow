"""

"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from typing import Union, Dict, List, Tuple

from ..dynamic_pca import tf_PCA # two usages only


bce_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, label_smoothing=0.0, reduction="auto", name="binary_crossentropy"
       )

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
                print("Stopping, loss change was under min_rel_change. Only adding last value")
            break
    if not scikit:
        pca.n_components_ = encoded.shape[-1]
        latent = pca.transform(encoded) # decoder has inverse transform 
    else:
        latent = encoded # lets assume perfect inverse transform here
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
    
    data and data_size can be passed to calculate and plot an ideal loss on the original data.
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


    