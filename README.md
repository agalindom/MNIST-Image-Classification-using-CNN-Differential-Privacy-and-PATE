## OVERVIEW
The notebook contains code for a Deep Learning image classification task using the concept of differential privacy and Private Aggregation of Teacher Ensembles (PATE) analysis and the pytorch library.

## Description
For the classification I used CNN for the network architecture, so in essence it is a typical approach for the MNIST dataset, what changed is in how the differential privacy was applied, there were a number of steps:

  - First, we divide of dataset into train and test where train would be the "labeled data" that contains sensitive information.

  - Second, divide the training dataset into n different teachers (where the "sensitive data is located") and train each different subset to get a different model from each.

  - Third, predict on our local data from each of the trained teacher models to later aggregate the predictions into a single variable containing the "n" teacher predictions and use them as labels for our unlabeled data.

  - Fourth, apply laplacian noise to each of the new labels obtained from the aggregated teacher predictions, to then use the PATE function to get the independent epsilon and the dependent epsilon which will serve us to know how much information we are leaking

  - Lastly, use this new labels on our "unlabeled" dataset and retrieve our test accuracy

## Files
  * Differential analysis and PATE.ipynb: jupyter notebook containing the code for the DL differential privacy analysis.

## Libraries
  * pytorch
  * pysyft
  * matplotlib
  * numpy
