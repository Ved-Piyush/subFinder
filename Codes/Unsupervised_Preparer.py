## In this script, we will look at the data preparation steps with the unsupervised sequences


## make all the library imports here
import pandas as pd

## function to take the many text files that are there in the 
## unsupervised folders, and create a nice csv after reading them in
def read_the_data(file):
    # read one file
    data = pd.read_csv(file, sep = " ", header = None)
    return data