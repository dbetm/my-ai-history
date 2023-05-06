# Dimensionality Reduction

## Principal Component Analysis (PCA)

**Applications**
- Noise filtering
- Visualization
- Feature extraction
- Stock market predictions
- Gene data analysis

**Objective**

Reduce the dimensions of a d-dimensional dataset by projecting it onto a (k)-dimentional subspace (where k<d).

**Algo**

1) Standardize the data.
2) Obtain the Eigenvectors and Eigenvalues from the covariance matrix or correlation matrix, or perform Singular Vector Decomposition.
3) Sort eigenvalues in descending order and choose the k eigenvectors that correspond to the k-largest eigenvalues where k is the number of dimensions of the new feature subspace (k <= d).
4) Construct the projection matrix W from the selected k eigenvectors.
5) Transform the original dataset X via W to obtain a k-dimensional feature subspace Y.

