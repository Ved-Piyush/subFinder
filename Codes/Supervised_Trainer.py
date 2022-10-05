from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multiclass import OneVsRestClassifier
# from Codes.embedding_modules import doc2vec_dm, doc2vec_dbow, word2vec_cbow, word2vec_sg, fasttext_sg, fasttext_cbow


def run_end_to_end(top_k, data, featurizer, K, known_unknown, model = None): 
    
    select_classes = data["high_level_substr"].value_counts()[:top_k].keys().tolist()
    known_unknown_data =  data[~data["high_level_substr"].isin(select_classes)]
    data = data[data["high_level_substr"].isin(select_classes)]
    data = pd.concat([data[["sig_gene_seq", "high_level_substr"]]], ignore_index = True)
    
    parameters_one_vs_rest = {"vr__n_estimators": [100, 300, 500]}
    
    params_best = []
    
    order = list(data["high_level_substr"].value_counts().index)
    
    
    NoneType = type(None)
    
    if type(model) != NoneType:
        vec_size = model.wv.vectors.shape[1]
    
    if known_unknown == True:
        data = pd.concat([data[["sig_gene_seq", "high_level_substr"]], 
                          known_unknown_data[["sig_gene_seq", "high_level_substr"]]],
                         ignore_index = True)
        
    skf_outer = StratifiedKFold(n_splits=K, random_state=42, shuffle = True)
    
    cm_all = np.zeros((len(order), len(order)))
    
    unraveled_positions = []
    overall_acc_list = []
    report_over_k = np.zeros((3, len(order)))
    avg_class_acc_k_list = []
#     avg_class_acc_k = 0
        
    for train_index, test_index in skf_outer.split(data["sig_gene_seq"],
                                              data["high_level_substr"].values):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
        class_weights = dict(1/(X_train["high_level_substr"].value_counts()/ X_train["high_level_substr"].value_counts().sum()))
        
        
        if featurizer == "countvectorizer":
        
            clf_one_vs_rest = Pipeline([('vectorizer',CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), 
                                                              lowercase = False)), 
                                
                                ('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                  ])
            
            gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 0)
            
        elif featurizer in ["doc2vec_dbow", "doc2vec_dm"]:
            
            X_train_doc_vectors = []
    
            for train_item in X_train["sig_gene_seq"].values:
                train_item = train_item.replace("|", ",").split(",")
                X_train_doc_vectors.append(model.infer_vector(train_item).tolist())
    
            X_test_doc_vectors = []
    
            for test_item in X_test["sig_gene_seq"].values:
                test_item = test_item.replace("|", ",").split(",")
                X_test_doc_vectors.append(model.infer_vector(test_item).tolist())   
                
            clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7, class_weight = class_weights))
                                                    ])
            gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 0)
                        
            
                
                
        elif featurizer in ["word2vec_cbow", "word2vec_sg", "fasttext_cbow", "fasttext_sg"]: 
            vocab = model.wv.index_to_key
            X_train_doc_vectors = []
    
            for train_item in X_train["sig_gene_seq"].values:
                train_item = train_item.replace("|", ",").split(",")
                word_vectors = []
                for word in train_item: 
                    if word in vocab:
                        word_vectors.append(model.wv.get_vector(word).reshape(1,-1).tolist()[0])
                    else:
                        word_vectors.append(np.zeros((1,vec_size)).reshape(1,-1).tolist()[0])
                
                
                if len(word_vectors) == 0: 
                    X_train_doc_vectors.append(np.zeros((1,vec_size)).tolist()[0])
                else:
                    X_train_doc_vectors.append(np.array(word_vectors).mean(0).tolist())    

            X_test_doc_vectors = []
    
            for test_item in X_test["sig_gene_seq"].values:
                test_item = test_item.replace("|", ",").split(",")
                word_vectors = []
                for word in test_item: 
                    if word in vocab:
                        word_vectors.append(model.wv.get_vector(word).reshape(1,-1).tolist()[0])
                    else:
                        word_vectors.append(np.zeros((1,vec_size)).reshape(1,-1).tolist()[0])
                        
                if len(word_vectors) == 0: 
                    X_test_doc_vectors.append(np.zeros((1,vec_size)).tolist()[0])
                else:
                    X_test_doc_vectors.append(np.array(word_vectors).mean(0).tolist())    
                    
            clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7, class_weight = class_weights))
                                                ])
            gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 0)
                    
        
        else:
            pass     
        

        if featurizer == "countvectorizer":
            gs_one_vs_rest.fit(X_train["sig_gene_seq"].values, X_train["high_level_substr"].values)
#             print(gs_one_vs_rest.best_params_)
#             print(gs_one_vs_rest.best_score_)
            y_test_pred = gs_one_vs_rest.predict(X_test["sig_gene_seq"].values)
        
        elif featurizer in ["doc2vec_dbow", "doc2vec_dm", "word2vec_cbow", "word2vec_sg", "fasttext_cbow", "fasttext_sg"]:
            gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["high_level_substr"].values)
#             print(gs_one_vs_rest.best_params_)
#             print(gs_one_vs_rest.best_score_)
            y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
            
        params_best.append(gs_one_vs_rest.best_params_)
        
        cm = confusion_matrix(X_test["high_level_substr"], y_test_pred, labels = order, normalize = 'true')
        cm_all+= cm
        
        unraveled_positions.append(cm.ravel().tolist())
        
        overall_acc_list.append(accuracy_score(X_test["high_level_substr"], y_test_pred))
        
        report = pd.DataFrame(classification_report(X_test["high_level_substr"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
        report_over_k += np.array(report)
        
        
        avg_class_acc = np.mean(np.diag(cm))
#         avg_class_acc_k += avg_class_acc
        
        avg_class_acc_k_list.append(avg_class_acc)
        
#         std_class_acc = 
        
    # average accuracy
    avg_acc = np.mean(overall_acc_list)

    df_cm = cm_all/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(df_cm, index = order,
                  columns = order)
    
    # avg classwise acc
    avg_class_acc = np.mean(avg_class_acc_k_list)
    
    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15)
    # plt.show()
    
    # average class 
#     avg_of_avg_class_acc_per_fold = np.mean(np.diag(df_cm))
#     std_avg_class_acc = np.std(np.diag(df_cm))

    flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    df_cm_std = pd.DataFrame(df_cm_std, index = order,
                  columns = order)

    # make the plot
    fig2 = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    plt.xlabel("Predicted Label", fontsize = 20)
    plt.ylabel("True Label", fontsize = 20)
    # plt.show()


    std_err_avg_acc = np.std(overall_acc_list)
    
    std_err_avg_classwise_acc = np.std(avg_class_acc_k_list)

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # make the plot
    fig3 = plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    
    #     overall_report

    overall_report = overall_report.mean(1)
    
    
    return avg_acc, avg_class_acc, std_err_avg_acc, std_err_avg_classwise_acc, overall_report, model, params_best, \
           fig, fig2, fig3
