"""
@author Daniel Sperber

This file contains some specialized but also general purpose
functions for plotting and saving pyplots

NOTE: This currently overwrites matplotlibs default style
to show some more axes and to be less plain.

NOTE2: Needs matplotlib 3.3+ for some methods
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from typing import List, Dict, Union, Tuple

# Optional packages
# import imageio # gif creation

# =============================================================================
# %%Automatically set Style
# =============================================================================
LOAD_STYLE = True

mysnsstyle = sns.axes_style('darkgrid', rc={'xtick.bottom': True,
                                            'ytick.left'  : True,
                                            'axes.grid'   : True,
                                            'axes.edgecolor' : '.25',
                                            'xtick.color': '.0',
                                            'ytick.color': '.0',
                                            
                                            })


COLORS   = ('red', 'blue', 'lightgreen', 'gray', 'cyan', 'yellow', 'purple')
_setxlabel = plt.xlabel # backup


def fontscale(s : float =1.7, legend_size : float=None) -> None:
    sns.set_context('notebook', font_scale=s, 
                    rc={'legend.fontsize': legend_size} if legend_size else None)

def load_style():
    sns.set_style(mysnsstyle)
    sns.set_context('notebook', font_scale=1.7, rc={'legend.fontsize': 17, 
                                                    "lines.linewidth": 2,  # from 1.5
                                                    })
    # Overwrite latex style
    matplotlib.rcParams['mathtext.fontset'] = 'dejavuserif'  # nicer delta

    #matplotlib.rcParams['mathtext.fontset'] = 'custom'
    #matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    #matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    #matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


if LOAD_STYLE and not "-default" in sys.argv:
    # This allows for some hackery to not overwrite the style.
    load_style()
    print("Overwriting pyplots default style.")


def global_xlog(enable=True):
    # some hackery to not always use plt.xscale('log')
    # might be some global function via context I haven't found yet.
    # so this is not 100% foolproof, especially not when using axes.
    if enable:
        def loglabels(*args, **kwargs):
            plt.xscale('log')
            _setxlabel(*args, **kwargs)
        plt.xlabel = loglabels
    else:
        plt.xlabel = _setxlabel

def reset_style():
    """
    Should reset my custom style.
    """
    from importlib import reload
    reload(sns)
    reload(matplotlib)

# %%=========== Saving ========================================================

def compare_svgs(fig, path, **kwargs):
    """
    Compares a pyplot figure to a saved .svg file.
    
    SVG files can not easily compared if they match, 
    because of ids and timestamp that differ even if identical.
    
    This tries to filter out certain lines that are unimportant
    todo improvable?
    """
    from io import StringIO
    buf = StringIO()
    fig.savefig(buf, format='svg', **kwargs)
    buf.seek(0)
    with open(path, "r") as f2:
        for line1, line2 in zip(buf, f2):
            if line1.startswith("    <dc:date>2"):
                continue
            if "clip-path=" in line1[:23]:
                idx = line1.find(" d=")
                line1 = line1[idx:]
                line2 = line2[idx:]
            elif line1.startswith('" id'):
                idx = line1.find("style=")
                line1 = line1[idx:]
                line2 = line2[idx:]
            elif "<use style" in line1[:20]:
                idx = line1.find("xlink:href=")
                idx2 = line1.find("y=")
                line1 = line1[:idx] + line1[idx2:]
                line2 = line2[:idx] + line2[idx2:]
            elif line1.startswith("  <clipPath id"):
                continue
                
            if line1 != line2:
                print("new:", line1)
                print("old:", line2)
                return False
    return True
        

def save_plot(path, fig=None, format='svg', check_exists=True, bbox_inches='tight', **kwargs):
    """
    Check if file already exists. Prompts to overwrite if not check_exists is
    False.
    **kwargs is passed to pyplots savefig.
    Useful keyword arguments can be:
        bbox_inches='tight',
    Here the format default is svg and not png.
    """
    from os.path import exists
    assert ":" not in path[2:], "No colons : in path" # this actually throws no error if saved, dangerous on Windows
    #import filecmp #NOTE: Filecmp does not work for svg as timestamp is written into it
    if fig is None:
        print("[INFO] No figure was passed. Using plt.savefig")
        fig = plt
    overwrite = None
    if exists(path) and check_exists:
        if format == 'svg' and compare_svgs(fig, path, **kwargs):
            overwrite = False # File matches
        else:
            import filecmp
            fig.savefig("temp."+format, format=format, bbox_inches=bbox_inches, **kwargs)
            if filecmp.cmp(path, "temp."+format):
                overwrite = False
        if overwrite is None:
            overwrite = input("Overwrite " + path +"\n (y)es / (n)o? ")
    else:
        overwrite = 'y'
    if overwrite == 'y':
        if not path.endswith(format):
            print("Info: Filename doesn't end with format", format, " changing it")
            path += format
            # todo fix wrong endings
        fig.savefig(path, format=format, bbox_inches=bbox_inches, **kwargs)

# %%=========== Heatmap =======================================================  

def matshow(m, cmap='coolwarm', **kwargs):
    """
    Uses searborn.heatmap, 
    Usefull kwargs:
        cmap (default coolwarm)
        vmin, vmax, x/yticklabels, annot=False
        center (for colorbar)
        cbar_kws={"ticks": [-1, 0, 1]} to adjust the colorbar.
    https://seaborn.pydata.org/generated/seaborn.heatmap.html
    """
    kwargs.setdefault("annot", True)
    kwargs.setdefault("linewidth", 0.5)
    return sns.heatmap(m, **kwargs)
   
# %%=========== Main ==================================================================

def plot_model_history(model_history):
    """
    Plots the history of a tensorflow model.fit training
    
    NOTE: Needs accuracy metric.
    Based upon:
    http://parneetk.github.io/blog/cnn-mnist/
    """
    plt.style.use("ggplot")
    fig, axs = plt.subplots(1,2,figsize=(10, 5))
    
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylim((0,1))
    
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1))
    axs[0].legend(['train_acc', 'val_acc'], loc='best')
    
    
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1))
    axs[1].legend(['train_loss', 'val_loss'], loc='best')
    plt.show()
    return fig
  



def _figs_mat_gen(figs, close=True):
    # Maybe with a generator this is more memory friendly, depends on imageio
    for fig in figs:
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img_mat = np.frombuffer(fig.canvas.print_to_buffer()[0], dtype='uint8')
        img_mat = img_mat.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # rgba
        if close:
            plt.close(fig)
        yield img_mat

def figs_to_gif(figs, output_path="./temp.gif", fps=0.25, closefigs=True):
    """
    Combines multiple figures to a gif
    """
    import imageio # optional
    res = imageio.mimsave(output_path, _figs_mat_gen(figs, closefigs), fps=fps)
    return res
    

  
# MNIST DIGITS
# -*- coding: utf-8 -*-
"""
NOTE: Needs matplotlib 3.3+ 
"""

def plot_matrix(mat, title= None, cmap='viridis', figsize=(8,6), show=True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.matshow(mat, cmap=cmap)
    ax.set_title(title)
    if show:
        plt.show()
    return fig


def ax_plot_digit(ax, digit, shape=(28, 28, -1), colormap=plt.cm.gray, aspect=None, xy_visible=(False, False)) -> None:
    #NOTE: Needs matplotlib 3.3+ for 28x28x1 to work!
    # Below shape is invalid errors.
    assert len(xy_visible) == 2, "xy must be an iterable of len 2" # does not catch data type errors
    ax.imshow(digit.reshape(shape), cmap=colormap, aspect=aspect)#extent=[-4,4,-1,1], )
    ax.get_xaxis().set_visible(xy_visible[0])
    ax.get_yaxis().set_visible(xy_visible[1])

def plot_image(data, shape=(28, 28, -1), show=True, cmap=plt.cm.gray) -> plt.Figure:
    fig, ax = plt.subplots()
    ax_plot_digit(ax, data, shape, cmap)
    if show:
        plt.show()
    return fig


def plot_wrong_classifications(XTest, YTest, Ypredicted, extra_text="", show=True) -> plt.Figure:
    """
    For a classification model
    Plots a sample of missclassified outputs
    """
    diff = np.where(YTest != Ypredicted)[0]
    print("Amount of wrongly predicted:", len(diff))
    
    fig, axes = plt.subplots(4, 9, figsize=(30,10))
    fig.suptitle("Sample of missclassified images\n" + extra_text)
    for ax, idx in zip(axes.flatten(), diff):
        ax_plot_digit(ax, XTest[idx])
        ax.set_title(str(YTest[idx]) + " classified as " + str(Ypredicted[idx]))
    if show:
        plt.show()
    return fig

def sort_outputs(original : np.array, labels : np.array, decoded : np.array =None, encoded : np.array =None) \
    -> Tuple[np.array,np.array,Union[np.array, None],Union[np.array, None], np.array]:
    """
    Sorts the outputs by their labels.
    Also returns the index positions where the outputs changed.
    """
    Y = labels.flatten()
    sort_indices = np.argsort(Y)
    Y = Y[sort_indices]
    return original[sort_indices], Y, (decoded[sort_indices] if decoded is not None else None), (encoded[sort_indices] if encoded is not None else None), np.unique(Y.flatten(), return_index=True)[1] 
    

def sample_class_indices(labels):
    """
    Returns indices where to take a single sample of each label
    """
    # Sort labels
    labels = np.array(labels).flatten()
    first_idx = np.unique(labels, return_index=True)[1] # Starting indices of labels
    return first_idx
    

def sample_classes(labels, sample_size=1, *args):
    """
    Retrieves (sample_size) samples from each label.
    from the args argument
    """
    labels = np.array(labels).flatten()
    args = list(args)               # for assignment as it is a tuple
    if sample_size > 0:
        # Sort by labels
        sorter = np.argsort(labels) 
        labels = labels[sorter]
        # sort args by sorted labels [0,0,0,...,1,1,1...]
        for i in range(len(args)):
            args[i] = np.array(args[i])[sorter]

    first_idx = np.unique(labels, return_index=True)[1] # Starting indices of labels
    samples = []
    if len(args) == 1:
        return args[0][sorted(np.r_[first_idx, first_idx+sample_size])] # maybe there is a direct way to not sort.
    for c in args:
        samples.append(c[np.r_[first_idx, first_idx+sample_size]]) # todo this does not give back multiple samples
    return samples
        

def plot_classes(X, Y, decoded=None, encoded=None, 
                 max_classes=10, 
                 figtext="", 
                 figtitle =None,
                 figtitle2=None,
                 colormap='magma', # for encoded
                 encoded_shape=(1, -1),):
    """
    Plots all classes with their original and
    optional decoded and encoded results.
    """
    # Sort by Y
    X, Y, decoded, encoded, class_start_indices = sort_outputs(X, Y, decoded, encoded)
    
    plot_n_classes = min(len(class_start_indices), max_classes) # Images tend to get to small
    #plt.style.use("default") # TODO
    naxes = 2
    if encoded is not None:
        naxes += 2
    if decoded is not None:
        naxes += 2
    fig, axes = plt.subplots(naxes, plot_n_classes, figsize=(11,8)) # plot 3x2 or 2x2
    if plot_n_classes == 1:
        axes.shape = (naxes, 1)
    if figtitle:
        fig.suptitle(figtitle)
    if figtitle2:
        fig.title(figtitle2())
        
    for i in range(plot_n_classes):
        ax = axes[0, i]
        ax_plot_digit(ax, X[class_start_indices[i]]) # Original
        ax_plot_digit(axes[1, i], X[class_start_indices[i]+1]) # Plot second sample
        
        if encoded is not None:
            ax = axes[2,i]
            ax_plot_digit(ax, encoded[class_start_indices[i]], 
                          shape=encoded_shape, colormap=colormap, aspect='auto')
            ax_plot_digit(axes[3,i],  encoded[class_start_indices[i]+1], 
                          shape=encoded_shape, colormap=colormap, aspect='auto')
        if decoded is not None:
            ax = axes[4 if encoded is not None else 2, i]
            ax_plot_digit(ax, decoded[class_start_indices[i]])
            ax_plot_digit(axes[5 if encoded is not None else 3, i], decoded[class_start_indices[i]+1])
    
    if figtext:
        plt.figtext(0, -0.5, figtext)
    plt.show()
    return fig

# todo deprecate this for plot_classes
def plot_all_digits(XTest, encoded, decoded, sample_idx, ldim_shape=(1, -1), 
                    colormap='magma', figtext="", figtitle=None, max_classes=10):
    #sample_idx = np.unique(Y.flatten(), return_index=True)[1] 
    plot_n_numbers = min(len(sample_idx), max_classes) if max_classes else len(sample_idx) # Images tend to get to small
    plt.style.use("default")

    fig, axes = plt.subplots(6, plot_n_numbers, figsize=(11,8))
    if plot_n_numbers == 1:
        axes.shape = (6, 1)
    if figtitle:
        fig.suptitle(figtitle)
    for i in range(plot_n_numbers):
        ax = axes[0, i]
        ax_plot_digit(ax, XTest[sample_idx[i]]) # Original
        ax_plot_digit(axes[1,i], XTest[sample_idx[i]+1]) # Plot second sample
        
        ax = axes[2,i]
        ax_plot_digit(ax, encoded[sample_idx[i]], 
                      shape=ldim_shape, colormap=colormap, aspect='auto')
        ax_plot_digit(axes[3,i], encoded[sample_idx[i]+1], 
                      shape=ldim_shape, colormap=colormap, aspect='auto')
        
        ax = axes[4,i]
        ax_plot_digit(ax, decoded[sample_idx[i]])
        ax_plot_digit(axes[5,i], decoded[sample_idx[i]+1])
    
    if figtext:
        plt.figtext(0, -0.5, figtext)
    plt.show()
    return fig
        
    
# todo deprecate this for plot_classes
def plot_number_sample(XTest, encoded, decoded, sample_idx, ldim_shape, 
                       number=1, plot_n_numbers=10, colormap='magma'):
    plt.style.use("default")
    fig, axes = plt.subplots(6, plot_n_numbers, figsize=(11, 8))
    for i in range(plot_n_numbers):
        
        ax_plot_digit(axes[0,i], XTest[sample_idx[number]+i]) # Original
        ax_plot_digit(axes[1,i], XTest[sample_idx[number]+i+plot_n_numbers]) # Plot second sample
        
        ax = axes[2,i]
        ax_plot_digit(ax, encoded[sample_idx[number]+i], 
                      shape=ldim_shape, colormap=colormap)
        ax_plot_digit(axes[3,i], encoded[sample_idx[number]+i+plot_n_numbers], 
                      shape=ldim_shape, colormap=colormap)
        
        ax = axes[4,i]
        ax_plot_digit(ax, decoded[sample_idx[number]+i])
        ax_plot_digit(axes[5,i], decoded[sample_idx[number]+i+plot_n_numbers])
           
    plt.show()
    
# ==============================

def plot_TSNE(encoded, true_labels, show=True, title="", suptitle="", perplexity=25):
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns

    tsne = TSNE(perplexity=perplexity, n_iter=1000, random_state=1000)
    # Expensive.
    print("Generating TSNE, this will take a bit.")
    shape = np.shape(encoded)
    plot_data_2D = tsne.fit_transform(np.reshape(encoded, (shape[0], np.prod(shape[1:]))))
    
    # Using pandas and seaborn is better than pyplot_functions.scatter, 
    # somehow that is very slow.
    print("Generating plot...")
    tsne_result_df = pd.DataFrame({'tsne_1': plot_data_2D[:,0], 'tsne_2': plot_data_2D[:,1], 'label': true_labels})
    
    fig, ax = plt.subplots(1, figsize=(9, 9))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="deep", data=tsne_result_df, ax=ax, s=40)
    
    lim = (plot_data_2D.min()-5, plot_data_2D.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    ax.set_title(title)
    fig.suptitle(suptitle)
    
    if true_labels is not None:
        # Shilouette Score
        from sklearn.metrics import silhouette_score
        score = silhouette_score(plot_data_2D, true_labels)
        text = "Shilouette Score : " + format(score, ".3f")
        print(text)
        plt.figtext(0,0, text)
    else:
        score = None
    
    if show:
        plt.show()
            
    return fig, score

# ==============================    
# Advanced
# %%========== Specials ===================================================================

# For XTrack, YTrack arguments.
def plot_tacked_values(YKeyword : Union[List[str], str], XYTracked : List[Tuple[str, Tuple[list, list]]], log_scale=False):
    """
    Creates a figure plot of YValues for example a loss, based on 
    certain changing settings X for example batch_size.
    
    Multiple settings, batch_size, epochs, ... can be plotted. 
    YKeyword is the name of the Y values inside XYTracked
    Which has the following logic:
    list((setting_1_name, (X:list, Y:list)), (setting_2_name, (X2:list, Y2:list), ...))
    
    # todo or why did I not use tuple (x_name, y_name) as dict keys?
    """
    if type(YKeyword) != list:
        YKeywordlist = [YKeyword]
    else:
        YKeywordlist = YKeyword
    if len(XYTracked) > len(YKeywordlist):
        # Assuming only one YKeyword[0] for multiple XKeywords are tracked:
        # Expanding, to match length for zip   
        YKeywordlist= [YKeywordlist[0] for _ in XYTracked]

    figs = []
    plt.style.use("ggplot")
    
    for (setting_key, (X, Y)), value_key in zip(XYTracked, YKeywordlist):
        #if type(value_key) != tuple:
        #    value_key = (value_key, )
        
        fig, ax = plt.subplots(1, figsize=(9, 6))
        ax.plot(X, Y, 'b.', markersize=20, label=value_key)
        ax.set_xscale('log' if log_scale else 'linear')
        setting_key = str(setting_key).replace("'", "")
        setting_key = setting_key.replace("(", "").replace(")", "")
        setting_key = setting_key.replace(",", " &")
        ax.set_xlabel(setting_key)
        if "tuple not yet supported" or len(YKeywordlist) == 1 or YKeywordlist[0] == YKeywordlist[1]:
            # only set ax label if one keyword is tracked
            ax.set_ylabel(value_key)
            # Clean tuple string

            #ax.set_title(str(value_key) + " against " + setting_key)
        else:
            # This is the case is YKeywords is a tuple, #todo: not supported yet.
            ax.legend()
        
        if len(str(X[0])) > 5:
            ax.tick_params(axis='x', labelrotation=-90)
        
        figs.append(fig)
        
    return figs
