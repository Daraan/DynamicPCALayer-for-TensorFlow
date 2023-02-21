# DynamicPCALayer-for-TensorFlow

With the two PCA Layers provided here the dimension of the data can be reduced dynamically inside a Tensorflow model.

Also provided is a minimal working example in the jupyter notebook and a full working one with additional methods for finding a good value for the choice of the reduction.

The best choice is to do a pretraining, calculate the PCA with the encoded data,
choose an appropriate dimension and retrain the model.
The PCALayer has a **estimate_reduction** method for that.
It calculates the loss at the different dimension and uses the elbow method to find the point,
where an improvement of the dimension only yields a below average improvement.

More flexible is the **pca_tests** function that can evaluate multiple loss functions if wanted.
Further it can early stop the test if no relevant improvements can be seen.
This function does not suggest a reduction value by itself, but you can choose the end point for example
or use the elbow method as well.

TODO:
* Add further test and possibilities
* Make compatible with weights to be savable
* pca_tests speed can not be improved by skipping dimensions


Further credits to:
* scikit-learn and their PCA implementation
* rafaelvalle for his implementation of the elbow method https://stackoverflow.com/a/37121355
