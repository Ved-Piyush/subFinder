## basic libary imports
import pandas as pd
import gensim
import numpy as np
# from datetime import datetime


# date and time
# now = datetime.now()
# now1 = str(now)
# now1 = "_".join(now1.split(" "))

## In this script we will train the various embedding models
from Codes.embedding_modules import doc2vec_dm, doc2vec_dbow, word2vec_cbow, word2vec_sg, fasttext_sg, fasttext_cbow

## Now read the unsupervised data corpus
updated_data_unsupervised = pd.read_csv(r"Data//Output//Unsupervised//all_unsupervised.csv")
updated_data_unsupervised.columns = ["sig_gene_seq"]
# another round of drop duplicates
updated_data_unsupervised = updated_data_unsupervised.drop_duplicates()

## prepare the unsupervised data as gensim expects
gene_list = [str(seq).replace("|", ",").split(",") for seq in updated_data_unsupervised["sig_gene_seq"]]
gene_list_tagged = [gensim.models.doc2vec.TaggedDocument(seq_list, [i]) for i, seq_list in enumerate(gene_list)] 

## need an estimate of how many unique tokens the supervised corpus has
## so that the dimensionality of the vectors we are using for BOW and these embedding modules are same
sup_data_path = r"Data/Supervised_Sequences/dbCAN-PUL_07-01-2022.xlsx"
supervised_data = pd.read_excel(sup_data_path, sheet_name="Sheet1")

## count number of unique genes in the supervised dataset
vec_size = len(np.unique([gene for seq in supervised_data["cazymes_predicted_dbcan"] for gene in seq.replace("|", ",").split(",")]))

## train the modules

## doc2vec_dm
doc2vec_dm = doc2vec_dm(vector_size = vec_size)
doc2vec_dm.build_vocab(gene_list_tagged)
doc2vec_dm.train(gene_list_tagged, total_examples=doc2vec_dm.corpus_count, epochs=doc2vec_dm.epochs)
doc2vec_dm.save(r"Embedding_Models//" + "doc2vec_dm" )


## doc2vec_dbow
doc2vec_dbow = doc2vec_dbow(vector_size = vec_size)
doc2vec_dbow.build_vocab(gene_list_tagged)
doc2vec_dbow.train(gene_list_tagged, total_examples=doc2vec_dbow.corpus_count, epochs=doc2vec_dbow.epochs)
doc2vec_dbow.save(r"Embedding_Models//" + "doc2vec_dbow"  )

## word2vec_cbow
word2vec_cbow = word2vec_cbow(vector_size = vec_size)
word2vec_cbow.build_vocab(gene_list)
word2vec_cbow.train(gene_list, total_examples=word2vec_cbow.corpus_count, epochs=word2vec_cbow.epochs)
word2vec_cbow.save(r"Embedding_Models//" + "word2vec_cbow"  )

## word2vec_sg
word2vec_sg = word2vec_sg(vector_size = vec_size)
word2vec_sg.build_vocab(gene_list)
word2vec_sg.train(gene_list, total_examples=word2vec_sg.corpus_count, epochs=word2vec_sg.epochs)
word2vec_sg.save(r"Embedding_Models//" + "word2vec_sg"  )

## fasttext_sg
fasttext_sg = fasttext_sg(vector_size = vec_size)
fasttext_sg.build_vocab(gene_list)
fasttext_sg.train(gene_list, total_examples=fasttext_sg.corpus_count, epochs=fasttext_sg.epochs)
fasttext_sg.save(r"Embedding_Models//" + "fasttext_sg"  )

## fasttext_cbow
fasttext_cbow = fasttext_cbow(vector_size = vec_size)
fasttext_cbow.build_vocab(gene_list)
fasttext_cbow.train(gene_list, total_examples=fasttext_cbow.corpus_count, epochs=fasttext_cbow.epochs)
fasttext_cbow.save(r"Embedding_Models//" + "fasttext_cbow"  )

