# exp.-formation-enthalpies

This is the repo. to store the code for the project of predicting exp. formation enthalpies by Sheng Gong, Woo Hyun Chae and Runze Liu

The paper will come out soon!

**************************
High-throughput dataset

**************************
Data collection

1. First, download the materials project database with all materials with formation energy by the materials project API (https://materialsproject.org/open).

2. Second, download the SSUB dataset from https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat, and use the provided script to clean the SSUB dataset.

3. Then, download the IIT database from https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835, and use the provided script to clean the IIT dataset.

4. Finally use the script to combine the two datasets. The random seed to make training/test set split is 11.

**************************
Training procedures:

Details of how we train the machine learning models are provided in the scripts.

Generally, we follow a two-step procedure:

1. Determine the best hyper-parameters by splitting the training set (80% of the whole dataset) into new training and validation set (80%*80% and 80%*20%).

2. Use the original training set (80% of the whole dataset) and the found best hyper-parameters to train machine learning models and test models' performance by the test set (20% of the whole dataset).

For the training of CGCNN and ROOST, detailed descriptions of usage, parameters, and input format for the two packages are provided at:

CGCNN: https://github.com/txie-93/cgcnn

ROOST: https://github.com/CompRhys/roost
