"""
Different datasets that can be loaded.
Each data can be set up here with some preprocessing

NOTE: That these datasets also have a classification model
that is also loaded from the global path varuables declared in this file.
"""

from tensorflow.keras.models import load_model # for compare_model

# =============================================================================
# MNIST
# =============================================================================

MNIST_CLASSIFIER_PATH = r"MNISTClassifier/MNIST-0.17"

def get_mnist_data(*args):
    from tensorflow.keras.datasets import mnist # only load when needed
    # Load MNIST
    (XTrain, YTrain), (XTest, YTest) = mnist.load_data()

    # Norm to [0,1]
    # Predictive model was
    XTrain = XTrain.astype('float32') / 255 
    XTest = XTest.astype('float32') / 255
            
    # Expand dim in place
    XTrain.shape = (*XTrain.shape, 1)
    XTest.shape = (*XTest.shape, 1)
    
    try:
        compare_model = load_model(MNIST_CLASSIFIER_PATH)
    except:
        print("WARNING: MNIST has no compare_model defined.")
        compare_model = None
    return XTrain, YTrain, XTest, YTest, compare_model
 
    
# =============================================================================
# Shapes
# =============================================================================

SHAPES_FILE = r"Shapes50k-V5.csv"
SHAPES_CLASSIFIER = "ShapesV5-Classifier"

def _get_non_encoded_shapes(test_size=0.1, random_state=42, shape=(28, 28), path="ConicShapes/"):
    import pandas as pd # only load when needed
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(path + SHAPES_FILE, compression='gzip')
    X = df.iloc[:, 1:].to_numpy()
    Y = df['shape'].to_numpy()
    
    # Split data
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    # Correct data
    XTrain, XTest = XTrain / 255, XTest / 255
    XTest.shape = XTrain.shape = (-1, *shape, 1) # or use expand dim when adding numpy 
    
    return XTrain, XTest, YTrain, YTest

def _encode_shape_labels(YTrain, YTest):
    from sklearn.preprocessing import LabelEncoder
    # Strings to integer
    le = LabelEncoder()
    YTrain = le.fit_transform(YTrain)
    YTest = le.transform(YTest)
    
    return YTrain, YTest, le
    
  
def get_conic_shapes_data(test_size=0.1, random_state=42, shape=(28, 28), path="ConicShapes/"):
    # Load data
    XTrain, XTest, YTrain, YTest = _get_non_encoded_shapes(test_size, random_state, shape, path)
    # string to integer labels
    YTrain, YTest, _ = _encode_shape_labels(YTrain, YTest)
    
    compare_model = load_model(path + SHAPES_CLASSIFIER)
    
    return XTrain, YTrain, XTest, YTest, compare_model


def get_filtered_shapes(use_shapes=['triangle'], test_size=0.1, random_state=42, shape=(28, 28), path="ConicShapes/"):
    """
    use_shapes can be used with get_topic_data(topic, TopicArgs)
    """
    XTrain, XTest, YTrain, YTest = _get_non_encoded_shapes(test_size, random_state, shape, path)
    if type(use_shapes[0] == str):
        YTrain, YTest, label_enc = _encode_shape_labels(YTrain, YTest)
        use_shapes = label_enc.transform(use_shapes)
    
    import numpy as np
    test_mask = np.isin(YTest, use_shapes)
    train_mask = np.isin(YTrain, use_shapes)
    
    YTrain = YTrain[train_mask]
    YTest  = YTest[test_mask]
    XTrain = XTrain[train_mask]
    XTest  = XTest[test_mask]

    compare_model = load_model(path + SHAPES_CLASSIFIER)
    return XTrain, YTrain, XTest, YTest, compare_model


# =============================================================================
# rock_paper_scissors
# =============================================================================

def get_rock_paper_scissors_data(shuffle=True, seed=42):
    """
    Needs tensorflow_datasets and might updates to ipywidgets, jupyterlab.
    Note defualt with shuffle=True, there is tf.random.set_seed(42) is called
    """
    import tensorflow as tf
    from tensorflow_datasets.image_classification import RockPaperScissors as data
    #if shuffle and seed is not None:
    #    tf.random.set_seed(seed)
    #data = tfds.load("rock_paper_scissors", split=['train', 'test'], with_info=False, shuffle=shuffle)
    train, test = data().as_dataset().values()

    # TODO: these are currently not usable the way they are!


# =============================================================================
# CIFAR
# =============================================================================

def get_cifar100_data(label_mode="coarse", grayscale=False, shape=(28, 28), center_data=False):
    """
    `label_mode` must be one of `"fine"`, `"coarse"`
    """
    import tensorflow as tf
    
    (XTrain, YTrain), (XTest, YTest) = tf.keras.datasets.cifar100.load_data(label_mode)
    if shape != (32, 32):
        # (32, 32) is the original size
        # This one needs some memory, might spew warning
        reshaper = tf.keras.layers.experimental.preprocessing.Resizing(*shape, interpolation="bilinear")
        XTrain = reshaper(XTrain).numpy()
        XTest = reshaper(XTest).numpy()
    if grayscale:
        XTrain = XTrain.mean(axis=-1, dtype='float32')
        XTest = XTest.mean(axis=-1, dtype='float32')
        # Reshape without creating new arrays, 
        XTrain.shape = (*XTrain.shape, 1)
        XTest.shape = (*XTest.shape, 1)
    
    XTrain = XTrain.astype('float32',  copy=False) / 255 # in case of grayscale this only divides
    XTest = XTest.astype('float32',  copy=False) / 255
    
    if center_data:
        XTest -= XTest.std()
        XTrain -= XTrain.std()
        XTest = XTest / XTest.mean()
        XTrain = XTrain / XTrain.mean()
        
    
    # compare_model
    if grayscale:
        if label_mode == 'fine':
            print("WARNING Predictive model does not exist for grayscale + fine")
            compare_model = None
        else:
            compare_model = load_model(r"CIFAR100/CIFAR100-Coarse-gray-Classifier")
    else:
        if label_mode == 'fine':
            compare_model = load_model(r"CIFAR100/CIFAR100-Fine-color-Classifier")
        else:
            compare_model = load_model(r"CIFAR100/CIFAR100-Coarse-color-Classifier")
    
    return XTrain, YTrain, XTest, YTest, compare_model

# =============================================================================
# =============================================================================

topic_collection = {
    "MNIST"             : get_mnist_data,
    "MNIST DIGITS"      : get_mnist_data,
    "Triangles"         : lambda : get_filtered_shapes(),
    "Circles"           : lambda : get_filtered_shapes(['circle']),
    "Conic Shapes"      : get_conic_shapes_data,
    "Filtered Shapes"   : get_filtered_shapes,
    "Shapes"            : get_conic_shapes_data,          # V4+
    "Cifar100"          : get_cifar100_data,
    "Cifar100Fine"      : lambda : get_cifar100_data(label_mode='fine', grayscale=True), # Same as with TopicArg (fine)
    "Cifar100ColoredFine" : lambda : get_cifar100_data(label_mode='fine', grayscale=False), # Same as with TopicArg (fine)
    "RockPaperScissor"  : NotImplemented,
}



def get_topic_data(topic : str, topicargs=None):
    """
    Loads the Test and Train data and the compare_model 
    of the dataset with the given topic name
    from the topic_collection dict.
    """
    if topicargs is not None:
        return topic_collection[topic](*topicargs)
    return topic_collection[topic]()



