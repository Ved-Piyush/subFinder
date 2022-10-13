## In this script, we will look at the data preparation steps with the unsupervised sequences


## make all the library imports here
import pandas as pd
import os

## function to take the many text files that are there in the 
## unsupervised folders, and create a nice csv after reading them in
def read_the_data(file_path):
    # check the tsvs inside each folder
    files = os.listdir(file_path)
    # read one file
    data = [pd.read_csv(file_path + "\\" + file, sep = "\t", header = None) for file in files ]
    data = pd.concat(data)
    return data