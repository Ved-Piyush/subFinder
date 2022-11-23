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
# from Codes.embedding_modules import doc2vec_dm, doc2vec_dbow, word2vec_cbow, word2vec_sg, fasttext_sg, fasttext_cbow

## Now read the unsupervised data corpus
updated_data_unsupervised = pd.read_csv(r"Data//Output//Unsupervised_10_12//all_unsupervised.csv")
updated_data_unsupervised = updated_data_unsupervised.sample(frac = 1.0).reset_index(drop = True)


# updated_data_unsupervised.columns = ["sig_gene_seq"]
# another round of drop duplicates
# updated_data_unsupervised = updated_data_unsupervised.drop_duplicates()

## prepare the unsupervised data as gensim expects
gene_list = [str(seq).replace("|", ",").split(",") for seq in updated_data_unsupervised["sequence"]]
gene_list_tagged = [gensim.models.doc2vec.TaggedDocument(seq_list, [i]) for i, seq_list in enumerate(gene_list)] 

## need an estimate of how many unique tokens the supervised corpus has
## so that the dimensionality of the vectors we are using for BOW and these embedding modules are same
sup_data_path = r"Data/Supervised_Sequences/supervised_seq_with_null.tsv"
supervised_data = pd.read_csv(sup_data_path, sep = "\t", header = None)
supervised_data.columns = ["PULID", "sequence"]

supervised_with_unsupervised_seqs = pd.DataFrame(pd.concat([updated_data_unsupervised["sequence"], supervised_data["sequence"]], ignore_index = True))
supervised_with_unsupervised_seqs.columns = ["sequence"]
supervised_with_unsupervised_seqs["sequence"] = [seq.replace("|", ",").replace(",", " ") for seq in supervised_with_unsupervised_seqs["sequence"]]
# supervised_with_unsupervised_seqs["sequence"].to_csv(r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", header=None, index=None, sep=' ', mode='a')
np.savetxt(r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", supervised_with_unsupervised_seqs.values, fmt='%s')


## count number of unique genes in the supervised dataset
vec_size = len(np.unique([gene for seq in supervised_data["sequence"] for gene in seq.replace("|", ",").split(",")]))
vec_size = np.min((300, vec_size))

## train the modules

## doc2vec_dm
doc2vec_dbow = gensim.models.doc2vec.Doc2Vec(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, min_count=5, epochs=60, workers = 7, dm = 0, 
                                      dbow_words = 0, window = 7)
doc2vec_dbow.save(r"Embedding_Models_10_12//" + "doc2vec_dbow" )


## word2vec_cbow
word2vec_cbow = gensim.models.Word2Vec(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None,
                                           sg = 0, workers = 7, epochs=60)

word2vec_cbow.save(r"Embedding_Models_10_12//" + "word2vec_cbow")


## word2vec_sg
word2vec_sg = gensim.models.Word2Vec(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1,
                                           workers = 7, epochs=60)

word2vec_sg.save(r"Embedding_Models_10_12//" + "word2vec_sg")

## fasttext_cbow
fasttext_cbow = gensim.models.fasttext.FastText(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 0,
                                           workers = 6, epochs=60)

fasttext_cbow.save(r"Embedding_Models_10_12//" + "fasttext_cbow" )

## fasttext_sg
fasttext_sg = gensim.models.fasttext.FastText(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, window = 7, min_count = 5, max_vocab_size = None, sg = 1,
                                           workers = 6, epochs=60)

fasttext_sg.save(r"Embedding_Models_10_12//" + "fasttext_sg" )


## doc2vec_dbow
doc2vec_dm = gensim.models.doc2vec.Doc2Vec(corpus_file=r"Data//Output//Unsupervised_10_12//all_unsupervised_text.txt", 
                                           vector_size=vec_size, min_count=5, epochs=60, workers = 7, dm = 1, 
                                      dbow_words = 0, window = 7)

doc2vec_dm.save(r"Embedding_Models_10_12//" + "doc2vec_dm")