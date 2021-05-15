# exp.-formation-enthalpies

This is the repo. to store the code for the project of predicting exp. formation enthalpies by Sheng Gong, Woo Hyun Chae and Runze Liu

**************************
Data collection

1. First, download the materials project database with all materials with formation energy by the materials project API (https://materialsproject.org/open).

2. Second, download the SSUB dataset from https://github.com/wolverton-research-group/qmpy/blob/master/qmpy/data/thermodata/ssub.dat, and use the provided script to clean the SSUB dataset.

3. Then, download the IIT database from https://figshare.com/collections/Experimental_formation_enthalpies_for_intermetallic_phases_and_other_inorganic_compounds/3822835, and use the provided script to clean the IIT dataset.

4. Finally use the script to combine the two datasets. The random seed to make training/test set split is 11.
