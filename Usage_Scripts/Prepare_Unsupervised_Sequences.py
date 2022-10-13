## In this script, we will look at the data preparation steps with the unsupervised sequences

## make all the library imports here
from Codes.Unsupervised_Preparer import read_the_data
import os
from joblib import Parallel, delayed
import pandas as pd
from tqdm import tqdm

## path to the unsupervised corpus
path_unsupervised = r"Data\Unsupervised_Sequences_10_12"

## check what all is included in this path
unsupervised_folders = os.listdir(path_unsupervised)


## iterate over these folders and create nice csv files
for folder in tqdm(unsupervised_folders):
    
    # path of this folder
    current_path = path_unsupervised + "\\" + folder
    
    # all the individual files in the folder
    all_files = os.listdir(current_path)
    
    # read the files in parallely
    all_files_df = Parallel(n_jobs=6, verbose = 3, backend = "threading")(delayed(read_the_data)(current_path + "\\" + i) for i in all_files)
    
    # convert to a dataframe
    all_unsupervised = pd.concat(all_files_df, ignore_index = True)
    
    # give a column name
    all_unsupervised.columns = ["cgc_id","sequence"]
    
    # remove duplicates
    # all_unsupervised = all_unsupervised.drop_duplicates("sequence")
    
    all_unsupervised.to_csv(r"Data//Output//Unsupervised_10_12//" + folder + ".csv", index = False)
    

unsupervised_csv_paths = r"Data//Output//Unsupervised_10_12"

# collate all unsupervised csvs and make a unified csv
unsupervised_csvs = os.listdir(unsupervised_csv_paths)

# read all the csvs as a dataframe and put in a list
all_csvs_df = [pd.read_csv(unsupervised_csv_paths + "\\" + csv) for csv in unsupervised_csvs]

# combine into one unified dataframe
all_unsupervised = pd.concat(all_csvs_df, ignore_index = True)

# write it out as a csv
all_unsupervised.to_csv(unsupervised_csv_paths + "\\" + "all_unsupervised.csv", index = False)

## find the duplicated ones
duplicated_seqs = all_unsupervised[all_unsupervised["sequence"].isin(all_unsupervised["sequence"][all_unsupervised["sequence"].duplicated()])].sort_values("sequence").reset_index(drop = True)

## write duplicated ones
duplicated_seqs.to_csv(unsupervised_csv_paths + "\\" + "duplicated_unsupervised.csv", index = False)

## fraction
duplicated_seqs.shape[0]/all_unsupervised.shape[0]