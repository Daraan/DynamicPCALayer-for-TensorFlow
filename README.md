# DynamicPCALayer-for-TensorFlow

With the two PCA Layers here the dimension of the data can be reduced dynamically.

Provides a minimal working example in the jupyter notebook and a full working one.

The best choice is to do a pretraining, calculate the PCA with the encoded data,
choose an approriate dimension and retrain the model.
The PCALayer has a **estimate_reduction** method for that.
It calculates the loss at the different dimension and uses the elbow method to find the point
where an improvement of the dimension only yields a below average improvement.

More flexible is the **pca_tests** function that can evaluate multiple loss functions if wantes.
Further it can early stop the test if no relevant improvements can be seen.
This function does not suggest a reduction value by itself but you can choose the end point for example
or use the elblow method as well.

TODO:
* Make compatible with weights to be savable
* pca_tests speed can not be improved by skipping dimensions


Further credits to:
* scikit-learn and their pca implementation
* rafaelvalle for his implemenation of the elbow method https://stackoverflow.com/a/37121355
