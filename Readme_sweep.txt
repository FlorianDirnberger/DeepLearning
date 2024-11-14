This manual ensures not burning down the HPC and making ourselves very unpopular at DTU ;) 

Manual: How to set up parametrization with wandb sweep:

1. Set your personal wandb login key in an .env file (have a look in the gitignore)

2. In the model add the variable you want to paramterize in the init constructor 

3. add it to the sweep config as the examples show 


Sweep-variables: 

First of all we should do research about suitable parameters and decide on a strategy on how to minimze runs (even on hpc one run is ~2 min)

List the parameters here following the template 

parameter: [Interval_min, Interval_max], [Chosen_val1, chosen_val2, ... ] # Argumentation

#tbd here# 
