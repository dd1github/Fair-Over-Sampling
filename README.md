

# Fair-Over-Sampling
## Description
This repository provides source code and data to implement the Fair Over-Sampling (FOS) method for bias mitigation, as described in the paper, "Towards Bridging Algorithmic Fairness and Imbalanced Learning." 
## Dependencies
The source code is built on IBM's AIF360 toolkit, which can be found at [AIF360](https://github.com/Trusted-AI/AIF360).  We recommend that you follow the instructions at AIF360 to either pip install the software or clone it in a separate conda environment.  AIF360 requires Tensorflow to implement certain of its features (we used TF ver. 2.6.0).
In addition, we used the following python libraries:
- Python v. 3.7.0
- Numpy v. 1.19.5
- Pandas v. 1.3.3
- Scikit Learn v. 0.24.2
## Data
We have included data to run FOS on the German Credit, Adult Census, and Compas Two-Year Recidivism datasets.  The data can be found in the data folder located in this repository. The data should be downloaded and placed in the ../data/ folder.
The orginal datasets can be found at:
- https://archive.ics.uci.edu/ml/datasets/adult
- https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
- https://github.com/propublica/compas-analysis
## How to Run Fair Over-Sampling
We have included a version of Fair Over-Sampling (FOS) that is intended to be used with standard classifiers (e.g., scikit learn's Support Vector Machines or Logistic Regression).  The main file for running FOS is FOS_main.py.  
The basic steps to run FOS are:
1. Select the AIF360 dataset that you would like to run (e.g., Adult Census, German Credit, or Compas) by commenting or uncommenting the respective lines in FOS_main.py.
2. Select a classifier (e.g., SVM or LG).
3. Input a link to the respective data folder that is saved on your local machine.
4. Run the file (FOS_main.py).

Related python files are:
1. Fair_OS.py contains the FOS algorithm.
2. common_utils.py generates useful metrics, including fair utility.


