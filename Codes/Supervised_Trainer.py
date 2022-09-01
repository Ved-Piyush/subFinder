## Do all the imports
## but probably need to do in the executable script

K = 10

from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
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
# from Codes.embedding_modules import doc2vec_dm, doc2vec_dbow, word2vec_cbow, word2vec_sg, fasttext_sg, fasttext_cbow



skf_outer = StratifiedKFold(n_splits=K, random_state=42, shuffle = True)

def balanced_random_forest_BOW(data, order): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    

    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vectorizer',CountVectorizer(tokenizer=lambda x: str(x).replace("|", ",").split(','), 
                                                              lowercase = False)), 
                                
                                ('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300, 500]}
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(X_train["cazymes_predicted_dbcan"].values, X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
    
    
    
        y_test_pred = gs_one_vs_rest.predict(X_test["cazymes_predicted_dbcan"].values)
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)  
    
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params


def balanced_random_forest_Doc2Vec_DM(data, order, trained_doc2vec_dm): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            X_train_doc_vectors.append(trained_doc2vec_dm.infer_vector(train_item).tolist())
    
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
           test_item = test_item.replace("|", ",").split(",")
           X_test_doc_vectors.append(trained_doc2vec_dm.infer_vector(test_item).tolist())    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300, 500]}
    
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)   
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params


def balanced_random_forest_Doc2Vec_DBOW(data, order, trained_doc2vec_dbow): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            X_train_doc_vectors.append(trained_doc2vec_dbow.infer_vector(train_item).tolist())
    
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
           test_item = test_item.replace("|", ",").split(",")
           X_test_doc_vectors.append(trained_doc2vec_dbow.infer_vector(test_item).tolist())    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300, 500]}
    
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)   
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params



def balanced_random_forest_Word2Vec_CBOW(data, order, trained_word2vec_cbow, vocab_cbow): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            word_vectors = []
            for word in train_item: 
                if len(vocab_cbow.intersection([word])) == 1:
                    word_vectors.append(trained_word2vec_cbow.wv.get_vector(word))
                
            if len(word_vectors) == 0: 
                X_train_doc_vectors.append(np.zeros((1,trained_word2vec_cbow.wv.vectors.shape[1])).tolist()[0])
        
            else:
                X_train_doc_vectors.append(np.array(word_vectors).mean(0).tolist())
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
            test_item = test_item.replace("|", ",").split(",")
            word_vectors = []
            for word in test_item: 
                if len(vocab_cbow.intersection([word])) == 1:
                    word_vectors.append(trained_word2vec_cbow.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_test_doc_vectors.append(np.zeros((1,trained_word2vec_cbow.wv.vectors.shape[1])).tolist()[0])
        
            else:
                X_test_doc_vectors.append(np.array(word_vectors).mean(0).tolist())
    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300,500]}
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)    
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params



def balanced_random_forest_Word2Vec_SG(data, order, trained_word2vec_sg, vocab_sg): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            word_vectors = []
            for word in train_item: 
                if len(vocab_sg.intersection([word])) == 1:
                    word_vectors.append(trained_word2vec_sg.wv.get_vector(word))
                
            if len(word_vectors) == 0: 
                X_train_doc_vectors.append(np.zeros((1,trained_word2vec_sg.wv.vectors.shape[1])).tolist()[0])
        
            else:
                X_train_doc_vectors.append(np.array(word_vectors).mean(0).tolist())
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
            test_item = test_item.replace("|", ",").split(",")
            word_vectors = []
            for word in test_item: 
                if len(vocab_sg.intersection([word])) == 1:
                    word_vectors.append(trained_word2vec_sg.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_test_doc_vectors.append(np.zeros((1,trained_word2vec_sg.wv.vectors.shape[1])).tolist()[0])
        
            else:
                X_test_doc_vectors.append(np.array(word_vectors).mean(0).tolist())
    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300,500]}
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)    
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params
    
def balanced_random_forest_FastText_SG(data, order, trained_fasttext_sg): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            word_vectors = []
            for word in train_item: 
                word_vectors.append(trained_fasttext_sg.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_train_doc_vectors.append(np.zeros((1,trained_fasttext_sg.wv.vectors.shape[1])).tolist()[0])
            else:
                X_train_doc_vectors.append(np.array(word_vectors).mean(0).tolist())   
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
            test_item = test_item.replace("|", ",").split(",")
            word_vectors = []
            for word in test_item: 
                word_vectors.append(trained_fasttext_sg.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_test_doc_vectors.append(np.zeros((1,trained_fasttext_sg.wv.vectors.shape[1])).tolist()[0])
            else:
                X_test_doc_vectors.append(np.array(word_vectors).mean(0).tolist())    
    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300,500]}
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)    
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params



def balanced_random_forest_FastText_CBOW(data, order, trained_fasttext_cbow): 
    cm_all = np.zeros((len(order), len(order)))
    unraveled_positions = []
    overall_acc = 0
    report_over_k = np.zeros((3, len(order)))
    best_params = []

    for train_index, test_index in tqdm(skf_outer.split(data["cazymes_predicted_dbcan"],
                                              data["updated_substrate (07/01/2022)"].values)):
        X_train, X_test = data.iloc[train_index,:], data.iloc[test_index,:]
    
    
        X_train_doc_vectors = []
    
        for train_item in X_train["cazymes_predicted_dbcan"].values:
            train_item = train_item.replace("|", ",").split(",")
            word_vectors = []
            for word in train_item: 
                word_vectors.append(trained_fasttext_cbow.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_train_doc_vectors.append(np.zeros((1,trained_fasttext_cbow.wv.vectors.shape[1])).tolist()[0])
            else:
                X_train_doc_vectors.append(np.array(word_vectors).mean(0).tolist())   
    
    
        X_test_doc_vectors = []
    
        for test_item in X_test["cazymes_predicted_dbcan"].values:
            test_item = test_item.replace("|", ",").split(",")
            word_vectors = []
            for word in test_item: 
                word_vectors.append(trained_fasttext_cbow.wv.get_vector(word))
            if len(word_vectors) == 0: 
                X_test_doc_vectors.append(np.zeros((1,trained_fasttext_cbow.wv.vectors.shape[1])).tolist()[0])
            else:
                X_test_doc_vectors.append(np.array(word_vectors).mean(0).tolist())    
    
    
    
        # class_weights = dict(1/(X_train["updated_substrate (07/01/2022)"].value_counts()/ X_train["updated_substrate (07/01/2022)"].value_counts().sum()))
    
        clf_one_vs_rest = Pipeline([('vr', BalancedRandomForestClassifier(n_jobs = 7))
                                    ])
    
        parameters_one_vs_rest = {"vr__n_estimators": [100,300,500]}
    

        gs_one_vs_rest = GridSearchCV(clf_one_vs_rest, parameters_one_vs_rest, cv = 5, n_jobs = 6, scoring = "balanced_accuracy", verbose = 10)
    
        gs_one_vs_rest.fit(np.array(X_train_doc_vectors), X_train["updated_substrate (07/01/2022)"].values)
    
        print(gs_one_vs_rest.best_params_)
        print(gs_one_vs_rest.best_score_)
        best_params.append(gs_one_vs_rest.best_params_)
    
        y_test_pred = gs_one_vs_rest.predict(np.array(X_test_doc_vectors))
    
    
        cm = confusion_matrix(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, normalize = 'true')
    
        cm_all+= cm
    
        unraveled_positions.append(cm.ravel().tolist())
    
        overall_acc += accuracy_score(X_test["updated_substrate (07/01/2022)"], y_test_pred)    
    
        report = pd.DataFrame(classification_report(X_test["updated_substrate (07/01/2022)"], y_test_pred, labels = order, output_dict  =True)).iloc[:3, :len(order)]
    
    
        report_over_k += np.array(report)    
        
        
    accuracy = overall_acc/skf_outer.get_n_splits()

    # dataframe for confusion matrix
    df_cm = pd.DataFrame(cm_all/skf_outer.get_n_splits(), index = order,
                  columns = order)

    # make the plot
    fig = plt.figure(figsize = (10, 10))
    sns.heatmap(df_cm, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    plt.title("10-fold averaged confusion matrix for the BOW BRF model", fontsize = 20, weight = "bold")
    plt.xlabel("Predicted Label",  weight = "bold", fontsize = 20)
    plt.ylabel("True Label", weight = "bold", fontsize = 20)
    plt.xticks(weight = "bold", fontsize = 15)
    plt.yticks(weight = "bold", fontsize = 15, rotation = 0)
    # plt.show()


    avg_accuracy = np.mean(np.diag(df_cm))

    avg_std_dev = np.std(np.diag(df_cm))

    # flattened_confusion_matrices = pd.DataFrame(unraveled_positions)
    # df_cm_std = np.array(flattened_confusion_matrices.std(0)).reshape(df_cm.shape[1],df_cm.shape[1])/np.sqrt(skf_outer.get_n_splits())
    # df_cm_std = pd.DataFrame(df_cm_std, index = order,
    #               columns = order)

    # # make the plot
    # fig1 = plt.figure(figsize = (10, 10))
    # sns.heatmap(df_cm_std, annot = True,  annot_kws={"fontsize":12, "weight":"bold"})
    # plt.title("Standard deviation for confusion matrix for the test set low level", fontsize = 20)
    # plt.xlabel("Predicted Label", fontsize = 20)
    # plt.ylabel("True Label", fontsize = 20)
    # # plt.show()


    # np.mean(np.diag(df_cm_std))

    overall_report = pd.DataFrame(report_over_k/skf_outer.get_n_splits())

    overall_report.columns = order

    overall_report.index = report.index

    # overall_report

    avg_overall_report = overall_report.mean(1)

    # make the plot
    fig1 =  plt.figure(figsize = (10, 10))
    sns.heatmap(overall_report, annot = True)
    plt.title("Classification Report", fontsize = 20)
    plt.ylabel("Metric Name", fontsize = 20)
    plt.xlabel("Substrate", fontsize = 20)
    # plt.show()
    
    return accuracy, avg_accuracy, avg_std_dev, avg_overall_report, fig, fig1, best_params
    