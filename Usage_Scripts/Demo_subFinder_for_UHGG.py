import pandas as pd
import gensim
import numpy as np
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier

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
avg_accuracy_bow, avg_std_dev_bow, avg_overall_report_bow

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
fig_word2vec_cbow 

## Precision Matrix
fig1_word2vec_cbow 

## best params
best_params_word2vec_cbow


## Now we have everything we need to get predictions for the UHGG data
## Step 1 - Get the UHGG data read in
## Step 2 - Encode the UHGG sequences using the winning embedding model
## Step 3 - Train the Balanced Random Forest Model using the best hyperparameter list
            # Basically this will be a list of 10 hyperparameters, for now we can choose the 
            # hyperparameter that appears most often in the list
## Step 4 - Get the substrate predictions for UHGG


## Step 1 
uhgg_data = pd.read_csv(r"Data/Output/Unsupervised/output_UHGG.csv")

## Step 2 get the winner embedding module
## winner was the word2vec module

trained_word2vec_cbow =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models//word2vec_cbow") 
vocab_cbow = set(trained_word2vec_cbow.wv.index_to_key)

## use the embedding module to convert the sequences to vectors
X_unsupervised_vectors = []
    
for train_item in tqdm(uhgg_data["sequence"].values):
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

## similarly also convert the supervised data sequences
X_supervised_vectors = []
    
for train_item in tqdm(data["cazymes_predicted_dbcan"].values):
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

## Now train and get predictions
catch_all_predictions = np.zeros((len(X_unsupervised_vectors), len(best_params_word2vec_cbow)))
catch_all_predictions = catch_all_predictions.astype(str)


# get the predictions
counter = 0
for param in tqdm(best_params_word2vec_cbow): 
    n_est = param["vr__n_estimators"]
    brf = BalancedRandomForestClassifier(random_state = 42, n_jobs = 7, n_estimators = n_est)
    brf.fit(X_supervised_vectors,  data["updated_substrate (07/01/2022)"])
    catch_all_predictions[:, counter] = brf.predict(X_unsupervised_vectors)
    counter = counter + 1

## take column wise modes and that will be the final prediction
from scipy import stats
mode_results = stats.mode(catch_all_predictions, axis = 1)
predictions_uhgg = mode_results.mode
uhgg_data["predicted_substrate"] = predictions_uhgg