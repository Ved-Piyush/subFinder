## In this script we will train supervised models
# library imports
import pandas as pd
import gensim
from Codes.Supervised_Trainer import balanced_random_forest_BOW, balanced_random_forest_Doc2Vec_DM, balanced_random_forest_Doc2Vec_DBOW,  \
                                     balanced_random_forest_Word2Vec_CBOW, balanced_random_forest_Word2Vec_SG, balanced_random_forest_FastText_SG, \
                                     balanced_random_forest_FastText_CBOW   

## read the data
sup_data_path = r"Data/Supervised_Sequences/dbCAN-PUL_07-01-2022.xlsx"
supervised_data = pd.read_excel(sup_data_path, sheet_name="Sheet1")

## removing the catch all classes
## for example multiple substrates and others 
old_data = pd.read_csv('Data/Supervised_Sequences/pul_seq_low_high_substr_year_corrected.tsv', sep = "\t")

old_data["high_level_substr"] = old_data["high_level_substr"].str.strip()

bad_puls = old_data[old_data["high_level_substr"].isin(["multiple_substrates", "mono/di/trisaccharide", "-", "human milk oligosaccharide", 
                                            "glycoprotein", "plant polysaccharide", "cellobiose"])]["PULid"].values

supervised_data = supervised_data[~supervised_data["ID"].isin(bad_puls)]

## top k
how_many = 5
order = list(supervised_data["updated_substrate (07/01/2022)"].value_counts()[:how_many].index)
data = supervised_data[supervised_data["updated_substrate (07/01/2022)"].isin(order)]
data_unknown = supervised_data[~supervised_data["updated_substrate (07/01/2022)"].isin(order)]
data_unknown["updated_substrate (07/01/2022)"] = "Others"
data = pd.concat([data, data_unknown], ignore_index = True)
order = list(data["updated_substrate (07/01/2022)"].value_counts().index)

## BOW model
accuracy_bow, avg_accuracy_bow, avg_std_dev_bow, avg_overall_report_bow, fig_bow, fig1_bow, best_params_bow = balanced_random_forest_BOW(data, order)


## Doc2Vec_DM model
## read the trained doc2vec DM model
trained_doc2vec_dm = gensim.models.doc2vec.Doc2Vec.load(r"Embedding_Models//doc2vec_dm") 
accuracy_doc2vec_dm, avg_accuracy_doc2vec_dm, avg_std_dev_doc2vec_dm, avg_overall_report_doc2vec_dm, fig_doc2vec_dm, fig1_doc2vec_dm, best_params_doc2vec_dm = balanced_random_forest_Doc2Vec_DM(data, order,trained_doc2vec_dm)


## Doc2Vec_DBOW model
## read the trained doc2vec DBOW model
trained_doc2vec_dbow = gensim.models.doc2vec.Doc2Vec.load(r"Embedding_Models//doc2vec_dbow") 
accuracy_doc2vec_dbow, avg_accuracy_doc2vec_dbow, avg_std_dev_doc2vec_dbow, avg_overall_report_doc2vec_dbow, fig_doc2vec_dbow, fig1_doc2vec_dbow, best_params_doc2vec_dbow = balanced_random_forest_Doc2Vec_DBOW(data, order,trained_doc2vec_dbow)

## Word2Vec_CBOW model
## read the trained Word2Vec_CBOW model
trained_word2vec_cbow =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models//word2vec_cbow") 
vocab_cbow = set(trained_word2vec_cbow.wv.index_to_key)
accuracy_word2vec_cbow, avg_accuracy_word2vec_cbow, avg_std_dev_word2vec_cbow, avg_overall_report_word2vec_cbow, fig_word2vec_cbow, fig1_word2vec_cbow, best_params_word2vec_cbow = balanced_random_forest_Word2Vec_CBOW(data, order,trained_word2vec_cbow, vocab_cbow)

## Word2Vec_SG model
## read the trained Word2Vec_SG model
trained_word2vec_sg =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models//word2vec_sg") 
vocab_sg = set(trained_word2vec_sg.wv.index_to_key)
accuracy_word2vec_sg, avg_accuracy_word2vec_sg, avg_std_dev_word2vec_sg, avg_overall_report_word2vec_sg, fig_word2vec_sg, fig1_word2vec_sg, best_params_word2vec_sg = balanced_random_forest_Word2Vec_SG(data, order,trained_word2vec_sg, vocab_sg)


## FastText_SG model
## read the trained FastText_SG model
trained_fasttext_sg =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models//fasttext_sg") 
accuracy_fasttext_sg, avg_accuracy_fasttext_sg, avg_std_dev_fasttext_sg, avg_overall_report_fasttext_sg, fig_fasttext_sg, fig1_fasttext_sg, best_params_fasttext_sg = balanced_random_forest_FastText_SG(data, order,trained_fasttext_sg)


## FastText_SG model
## read the trained FastText_SG model
trained_fasttext_cbow =gensim.models.word2vec.Word2Vec.load(r"Embedding_Models//fasttext_cbow") 
accuracy_fasttext_cbow, avg_accuracy_fasttext_cbow, avg_std_dev_fasttext_cbow, avg_overall_report_fasttext_cbow, fig_fasttext_cbow, fig1_fasttext_cbow, best_params_fasttext_cbow = balanced_random_forest_FastText_CBOW(data, order, trained_fasttext_cbow)


