import pandas as pd
import gensim
import numpy as np
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
from scipy.stats import binom
from sklearn.multiclass import OneVsRestClassifier
## In this script we will show how to use the created subFinder pipeline for making predictions for the UHGG data

## make sure the UHGG data is in the Data/Unsupervised_Sequences
## in the original format as the files were shared with us

## then run the following code
exec(open("Usage_Scripts/Prepare_Unsupervised_Sequences.py").read())

## this will create a file called all_unsupervised.csv in the Output/Unsupervised
## that file will be used to create the embedding models

## the following code will train all the embedding models
exec(open("Usage_Scripts/Train_Embedding_Modules.py").read())

## Now run the supervised training script
## which will take the annotated cgc files
## and train the supervised models
exec(open("Usage_Scripts/Train_Supervised_Models.py").read())


## now decide which embedding method was the best
## and get the best hyper-parameters for the BRF trained
## on that embedding method

## get the average accuracy for all the methods and the standard deviation of the classwise accuracies
## also other metrics like Precision, F1, Recall

## for BOW
accuracy_bow, avg_accuracy_bow, avg_overall_report_bow 

## for Doc2Vec_DM
avg_accuracy_doc2vec_dm, avg_std_dev_doc2vec_dm, avg_overall_report_doc2vec_dm

## for Doc2Vec_DBOW
avg_accuracy_doc2vec_dbow, avg_std_dev_doc2vec_dbow, avg_overall_report_doc2vec_dbow

## for Word2Vec_CBOW
avg_accuracy_word2vec_cbow, avg_std_dev_word2vec_cbow, avg_overall_report_word2vec_cbow

## for Word2Vec_SG
avg_accuracy_word2vec_sg, avg_std_dev_word2vec_sg, avg_overall_report_word2vec_sg

## for FastText_SG
avg_accuracy_fasttext_sg, avg_std_dev_fasttext_sg, avg_overall_report_fasttext_sg

## for FastText_CBOW
avg_accuracy_fasttext_cbow, avg_std_dev_fasttext_cbow, avg_overall_report_fasttext_cbow


###### Winner Winner #####
## winner winner chicken dinner is
avg_accuracy_word2vec_cbow, avg_std_dev_word2vec_cbow, avg_overall_report_word2vec_cbow

## confusion matrix 

fig_word2vec_cbow.savefig(r"Results/best_confusion_matrix")

## Precision Matrix
fig1_word2vec_cbow.savefig(r"Results/best_precision_matrix") 

## best params
model_lists_bow


## Now we have everything we need to get predictions for the UHGG data
## Step 1 - Get the UHGG data read in
## Step 2 - Encode the UHGG sequences using the winning embedding model
## Step 3 - Train the Balanced Random Forest Model using the best hyperparameter list
            # Basically this will be a list of 10 hyperparameters, for now we can choose the 
            # hyperparameter that appears most often in the list
## Step 4 - Get the substrate predictions for UHGG

# this data below is from train supervised models
## make this code change elsewhere so that it is end to end
data = pd.read_csv('Data/Supervised_Sequences/reduced_sup_data.csv')
data.columns = ['PUL ID', 'high_level_substr', 'PULid', 'sig_gene_seq']
data = data[data["high_level_substr"] != "capsule polysaccharide synthesis"]

to_keep = list(data["high_level_substr"].value_counts().index[:top_k])


## Step 1 
oral_data = pd.read_csv(r"Data/Output/Unsupervised_10_12/output_humanoral.csv")
oral_data = oral_data.drop_duplicates("sequence").reset_index(drop = True)

## Step 2 get the winner embedding module
## winner was the word2vec module

trained_word2vec_cbow =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models_10_12//word2vec_sg") 
vocab_cbow = set(trained_word2vec_cbow.wv.index_to_key)

## use the embedding module to convert the sequences to vectors
X_unsupervised_vectors = []
    
for train_item in tqdm(oral_data["sequence"].values):
    train_item = train_item.replace("|", ",").split(",")
    word_vectors = []
    for word in train_item: 
        if len(vocab_cbow.intersection([word])) == 1:
            word_vectors.append(trained_word2vec_cbow.wv.get_vector(word))
    if len(word_vectors) == 0: 
        X_unsupervised_vectors.append(np.zeros((1,trained_word2vec_cbow.wv.vectors.shape[1])).tolist()[0])
    else:
        X_unsupervised_vectors.append(np.array(word_vectors).mean(0).tolist())

X_unsupervised_vectors = np.array(X_unsupervised_vectors)


## Step 3

data = data[data["high_level_substr"].isin(to_keep)].reset_index(drop = True)


## similarly also convert the supervised data sequences
X_supervised_vectors = []
    
for train_item in tqdm(data["sig_gene_seq"].values):
    train_item = train_item.replace("|", ",").split(",")
    word_vectors = []
    for word in train_item: 
        if len(vocab_cbow.intersection([word])) == 1:
            word_vectors.append(trained_word2vec_cbow.wv.get_vector(word))
    if len(word_vectors) == 0: 
        X_supervised_vectors.append(np.zeros((1,trained_word2vec_cbow.wv.vectors.shape[1])).tolist()[0])
    else:
        X_supervised_vectors.append(np.array(word_vectors).mean(0).tolist())


X_supervised_vectors = np.array(X_supervised_vectors)

## To get easy to understand p values
## we need to change the model a bit

model_check =  OneVsRestClassifier(BalancedRandomForestClassifier(n_jobs = 7, class_weight = "balanced"))
model_check.fit(X_supervised_vectors,  data["high_level_substr"].values)
class_order = model_check.classes_

# get the predictions
preds_df = pd.DataFrame()
counter = 0
outer_catch = []

# we will store the normalized probabilities here
normalized_probs = np.zeros((len(X_unsupervised_vectors), len(class_order)))

## ensemble of 15 models
how_many = 15

## loop over
for inner in tqdm(range(0,how_many)): 
    
    ## one vs all model
    model =  OneVsRestClassifier(BalancedRandomForestClassifier(n_jobs = 7, class_weight = "balanced"))
    model.fit(X_supervised_vectors,  data["high_level_substr"].values)
    preds_full = model.predict(X_unsupervised_vectors)
    preds_proba = model.predict_proba(X_unsupervised_vectors)
    normalized_probs += preds_proba
    
    ## grab all the separate one vs all estimators
    ova_ests = model.estimators_
    inner_counter = 0
    inner_catch = []
    for ests in ova_ests: 
        preds = ests.predict_proba(X_unsupervised_vectors)     
        # preds_bin = ests.predict(X_unsupervised_vectors)   
        # preds_bin = pd.DataFrame(preds_bin)
        # preds_bin.columns = [class_order[inner_counter] + "_" + str(counter) + "_yes_or_no"]
        preds_df1 = pd.DataFrame(preds)
        preds_df1 = preds_df1[[1]]
        preds_df1 = pd.DataFrame(preds_df1)
        preds_df1.columns = ["unnormalized_prob_" + class_order[inner_counter] + "_" + str(counter)]
        preds_df1 = pd.concat([preds_df1], 1)
        inner_catch.append(preds_df1)
        # preds_df = pd.concat([preds_df, preds_df1], 1)
        inner_counter += 1
    
    inner_catch_df = pd.concat(inner_catch,1)
    outer_catch.append(inner_catch_df)
    counter = counter + 1


outer_catch_df = pd.concat(outer_catch,1)
reorder = np.sort(outer_catch_df.columns)
outer_catch_df = outer_catch_df[reorder]
outer_catch_df["sequence"] = oral_data["sequence"].values
cols = list(outer_catch_df.columns)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('sequence')))
outer_catch_df = outer_catch_df[cols]


normalized_probs/= how_many
normalized_probs = pd.DataFrame(normalized_probs)
normalized_probs.columns = ["normalized_prob_" + item  for item in class_order]
normalized_probs["sequence"] = oral_data["sequence"]
cols = list(normalized_probs.columns)
# move the column to head of list using index, pop and insert
cols.insert(0, cols.pop(cols.index('sequence')))
normalized_probs = normalized_probs[cols]
normalized_probs["predicted_substrate"] = normalized_probs.iloc[:,1:].idxmax(axis=1)
normalized_probs["predicted_substrate"] = normalized_probs["predicted_substrate"].map(lambda x: x.split("_")[-1])

outer_catch_df = pd.melt(outer_catch_df, id_vars = ["sequence"])
outer_catch_df = outer_catch_df.reset_index(drop = True)
outer_catch_df["substrate"] = outer_catch_df["variable"].map(lambda x: x.split("_")[-2])
outer_catch_df["order"] = outer_catch_df["variable"].map(lambda x: x.split("_")[-1])
outer_catch_df = outer_catch_df.sort_values(["sequence", "substrate"], ascending = [True, True])
outer_catch_df["yes_or_no"] = outer_catch_df["value"].map(lambda x: 1 if x > 0.5 else 0)
outer_catch_df_summary = outer_catch_df.groupby(["sequence", "substrate"]).aggregate({"value": np.mean, "yes_or_no": np.sum})
outer_catch_df_summary = outer_catch_df_summary.reset_index()
outer_catch_df_summary.columns = ["sequence", "substrate", "probability_score", "successes"]
# outer_catch_df = pd.concat([outer_catch_df, outer_catch_df_summary])

def p_value_function(successes, trials= how_many, prob = 0.5): 
    prob = binom.cdf(successes, trials, prob)
    return 1-prob

outer_catch_df_summary["p_value"] = outer_catch_df_summary["successes"].map(p_value_function)
outer_catch_df_summary = outer_catch_df_summary.drop("successes", 1)
outer_catch_df_summary = outer_catch_df_summary.sort_values(["sequence", "p_value"], ascending = [True, True])
outer_catch_df_summary = outer_catch_df_summary.reset_index(drop = True)

outer_catch_df_summary.to_csv("Data/Output/Predictions/Predictions_oral_with_probability_and_p_values_Blast_Style_new_sup_new_unsup.csv", index = False)
normalized_probs["predicted_substrate"].value_counts()