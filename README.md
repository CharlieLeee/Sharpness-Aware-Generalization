# OptML-Project
Repository containing the accompanying code to our optimization for machine learning paper "Model Generalization: A Sharpness Aware Optimization Perspective" 

## How to use this code
Clone the repository
```
git clone --recurse-submodules https://github.com/CharlieLeee/OptML-Project.git
```

Set up the anaconda environment with:
```
conda env create -n OptMLProject --file environment.yml
```

In order to replicate the experiments in the paper, the folders `local_scripts` and `izar_scripts` contain the shell scripts that were used to execute the experiments. These expeiments resulted in .csv files containing the results, which are stored in the `Notebooks/acc` and `Notebooks/loss` folders.

In order to recreate the plots used in the paper, the file `Notebooks/Generate plot in the report.ipynb` is used. 

## File structure

`izar_scripts` contains the scripts for experiment 1 and 2.

`local_scripts/` contain scripts to run the ASAM vs SAM comparison in experiment C.

`run.py` contains the code to run the experiments.

`Notebooks` contains the experiment data, and the notebook to generate the plots used in the paper.

## Resources
- A/SAM: https://github.com/davda54/sam
