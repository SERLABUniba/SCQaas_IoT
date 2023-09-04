#!/usr/bin/env python
# coding: utf-8

#################################################################
# Import libraries
#################################################################
import random
import string

# Modelling
import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
from IPython.display import clear_output
#import graphviz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
import sys
from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qboost.qboost import QBoostClassifier
from dwave.system.samplers import DWaveSampler
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dwave.system.composites import EmbeddingComposite
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn import metrics

import psycopg2
# from psycopg2.extras import RealDictCursor
from timeit import default_timer as timer
from datetime import *
from collections import Counter
from connect import connect

# Garbage collection
import gc

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from numpy import mean

from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import Counter

# Import Qiskit libraries
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.circuit.library import EfficientSU2

from qiskit.circuit.library import ZFeatureMap
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import PegasosQSVC

import seaborn as sns

# gc.collect()

# Globals Variables
# QBoost Parameters and Variables
#DW_PARAMS = {'num_reads': 3000,
#             'auto_scale': True,
#             # "answer_mode": "histogram",
#             'num_spin_reversal_transforms': 10,
#             # 'annealing_time': 10,
#             # 'postprocess': 'optimization',
#             }
#NUM_WEAK_CLASSIFIERS = 35
#TREE_DEPTH = 3
# dwave_sampler = DWaveSampler(token="DEV-98f903479d1e03bc59d7ba92376a492f76f7c906")  # anibrata.pal@uniba.it
#dwave_sampler = DWaveSampler(token="DEV-68adb3eb159f7d774b2744b76f46444f8faa1ac5")  # ahanamaitra01@gmail.com
# sa_sampler = micro.dimod.SimulatedAnnealingSampler()
#emb_sampler = EmbeddingComposite(dwave_sampler)
#lmd = 0.5

# Qiskit globals random seed
algorithm_globals.random_seed = 42

# Create empty array for callback to store evaluations of the objective function as global variable
objective_func_vals = []

# Path to the dataset
#path = '../../../Dataset/IoT Intrusion Dataset/IoT Network Intrusion Dataset.csv'

path1 = '../data/mqtt_train70_reduced.csv'
path2 = '../data/mqtt_test30_reduced.csv'

#####################################################################################
# Methods
#####################################################################################


def loadData(path):
    data = pd.read_csv(path)
    # print(data)
    return data


# Report results in files
def reportResult(y, predictions, algo, num_comp, predict_type, train_time, predict_time):
    print("Reporting Result ...")
    result_time = datetime.now().strftime("%Y%m%d%H%M%S")
    report = classification_report(y, predictions)
    with open('../results/' + str(result_time) + '_' + str(algo) + '_' + str(num_comp) + str(predict_type) + '_report.txt', 'a') as f:
        f.write('Training time :' + str(train_time) + '\n')
        f.write(report)
        f.write('Prediction time :' + str(predict_time) + '\n')
    f.close()
    print("Saved...")


# The function below finds out the quality metrics for different values of K(nearest neighbors for SMOTE) and
# oversampling strategies for SMOTE.
# The code below uses the QBoost to find out good parameters for oversampling, and undersampling
def evaluate_over_under(X_pca, y_pca):
    # Nearest neighbor values for SMOTE
    k_values = [1, 2, 3, 4, 5, 6, 7]
    s_values = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Create a KFold object with n_splits=10
    # kf = KFold(n_splits=10)
    # precision, recall, f1 = 0, 0, 0
    for s in s_values:
        for k in k_values:
            # Declare the models for SMOTE over and Random Undersampling
            over = SMOTE(sampling_strategy=s, k_neighbors=k)
            under = RandomUnderSampler(sampling_strategy=0.5)
            X, y = over.fit_resample(X_pca, y_pca)
            X, y = under.fit_resample(X, y)
            print(Counter(y))
            count = 0
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
            model, train_time = trainQboost(x_train, y_train)
            predictions, predict_time = predictModel(model, x_test)
            # print('Training Time: %.2f, Prediction Time: %2f' % (train_time, predict_time))
            # Calculate precision
            precision = precision_score(y_test, predictions)
            # Calculate recall
            recall = recall_score(y_test, predictions)
            # Calculate F1-score
            f1 = f1_score(y_test, predictions)
            print('>> s: %3f -> k: %d -> Precision: %.3f, Recall: %.3f, F1_score: %.3f, Training Time: %.2f, '
                  'Prediction Time: %2f' % (s, k, precision, recall, f1, train_time, predict_time))

    # Create a module to find out the best values of s and k and return those
    return s, k


def dataClean(df):
    # #### Find non-numeric features
    non_numeric_columns = df.select_dtypes(exclude=[int, float]).columns.tolist()
    print(non_numeric_columns)

    # Check
    df.head()
    print(df['target'].unique())
    # print(df['target'].value_counts()['Normal'])
    # print(df['target'].value_counts()['Anomaly'])

    # Remove the Target Variable / Dependent Variable from the non-numeric columns list
    # Label is non-numeric, but it is the dependent variable;
    # need for data processing, remove it from the non-numeric column list
    non_numeric_columns.remove('target')
    print(non_numeric_columns)

    # Remove non-numeric columns from the dataset, but keeping the 'Label'
    df = df.drop(columns=non_numeric_columns)

    # Change "Label" from text to numbers
    df['target'] = df['target'].replace('legitimate', -1).replace('dos', 1).replace('malformed', 1).\
        replace('bruteforce', 1). replace('slowite', 1). replace('flood', 1)

    # Check
    df['target'].unique()

    # Find out the datatypes of the columns in the dataframe
    column_info = df.dtypes
    print(column_info)

    # Find out the columns where all values are equal -> does not contribute to the result
    columns_with_same_val = df.columns[(df == df.iloc[0]).all()]
    print(columns_with_same_val)

    # Drop all columns with same values for all records
    df = df.drop(columns=columns_with_same_val)

    # Check
    df.head()

    return df


def impute(df):
    # Find out the max values from each column to avoid 'inf' -> infinity value problem
    print(df.max(axis=0))

    # Some columns above have Infinity value and cannot be processed as it is, so they need to be replaced with NaN
    # and then imputed

    # Find out the columns which have NaN / Missing values - None here
    nan_values = df.columns[df.isna().any()]
    print(nan_values)

    # Replacing Infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Imputing using median for all NaN values
    df.fillna(df.median(), inplace=True)

    return df


# Create/Separate independent and dependent data
def create_xy(df):
    # Separate Dependent and Independent columns
    shp = df.shape  # Shape of the dataframe
    cols = list(df.columns.values)  # List of features

    # Independent data
    X = df[cols[0:shp[1] - 1]]
    # Dependent data
    y = df['target']

    # Check
    X.head(), y.head()

    return X, y


# Apply PCA to use only some features among a large number of features
def apply_pca(X, classifier, num_comp):
    # Feature selection using PCA
    # Preprocess the data by scaling the features

    # Use MinMaxScaler
    if classifier == 'pegasos':
        scaled_X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    else:
        scaled_X = MinMaxScaler().fit_transform(X)

    # Check
    print(scaled_X)

    # Perform basic PCA for dataset analysis
    pca = PCA()
    pca.fit(scaled_X)

    # Plot the variance graph for finding the required number of principal components
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    #plt.show()

    elbow_index = elbow(explained_variance)
    threshold_index = threshold(explained_variance, cumulative_variance)

    if elbow_index <= threshold_index:
        n_components = elbow_index
    else:
        n_components = threshold_index

    # Override number of components for ease of running
    n_components = num_comp
    print('Override the number of components: ', n_components)

    # From the above-mentioned statistical analysis, we see that including more or less 20 components is enough
    # for a variance that levels off
    # Number of Components with impactful positive correlation = 20
    # n_components = 20  # Choose the desired number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(scaled_X)

    # Check
    print(X_pca)

    return X_pca


# Use Elbow method to find the optimal number of components
def elbow(explained_variance):
    # Finding the best number of components by Elbow method
    # Calculate the difference in explained variance between components
    explained_variance_diff = np.diff(explained_variance)

    # Find the index of the elbow point (maximum difference)
    elbow_index = np.argmax(explained_variance_diff) + 1

    print("Number of components at the elbow point:", elbow_index)

    return elbow_index


# Use Threshold method to find the optimal number of components
def threshold(explained_variance, cumulative_variance):
    # Find the best number of components by the Threshold method
    threshold_val = 0.9985  # Define the desired threshold (e.g., 99% variance explained)

    # Find the number of components above the threshold
    threshold_index = np.argmax(cumulative_variance >= threshold_val) + 1

    # Plot the threshold line
    plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, marker='o')
    plt.axhline(y=threshold_val, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    #plt.show()

    print("Number of components above the threshold:", threshold_index)

    return threshold_index

    # The Dataset used here is severely imbalanced, so we use a combination of Oversampling using K nearest neighbors
    # SMOTE Oversampling and Random Undersampling
    # The part below is for checking the recall based on the number of k neighbors used in SMOTE oversampling

    # The commented code below is a template used to test Over and under-sampling for DT classifier
    # values to evaluate
    # k_values = [1, 2, 3, 4, 5, 6, 7]
    # for k in k_values:
    #     # define pipeline
    #     model = DecisionTreeClassifier()
    #     over = SMOTE(sampling_strategy=0.1, k_neighbors=k)
    #     under = RandomUnderSampler(sampling_strategy=0.5)
    #     steps = [('over', over), ('under', under), ('model', model)]
    #     pipeline = Pipeline(steps=steps)
    #     # evaluate pipeline
    #     cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    #     scores = cross_val_score(pipeline, X_pca, y, scoring='recall', cv=cv, n_jobs=-1)
    #     score = mean(scores)
    #     print('> k=%d, Mean Recall: %.3f' % (k, score))

# Training Qboost
def trainQboost(X, y):
    print('Training QBOOST Model... ')
    qboost = QBoostClassifier(n_estimators=NUM_WEAK_CLASSIFIERS, max_depth=TREE_DEPTH)
    start = timer()
    qboost.fit(X, y, emb_sampler, lmd=lmd, **DW_PARAMS)
    end = timer()
    train_time = end - start
    # print('QBoost training time in seconds :', train_time)
    return qboost, train_time


# Declare VQC
def vqc_nat(feature_map, ansatz):
    return VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=30),
        callback=callback_graph,
    )


# Training VQC
def trainVQC(X, y, data_dimension):
    print('Training VQC Model... ')
    feature_map = ZZFeatureMap(data_dimension, reps=1)
    ansatz = RealAmplitudes(data_dimension, reps=3)
    vqc = vqc_nat(feature_map, ansatz)
    start = timer()
    # fit classifier to data
    vqc.fit(X, y)
    end = timer()
    train_time = end - start
    return vqc, train_time


# Train PegaSOS QSVC
def trainPQSVC(X, y, data_dimension):
    print('Training PegaSOS QSVC Model... ')
    # Number of qubits is equal to the number of features
    num_qubits = data_dimension  # Test with high data dimension to check if this works
    # Number of steps performed during the training procedure
    tau = 100  # Test with other number of steps to check difference in data
    # Regularization parameter
    C = 1000

    # Change the data for PegaSoS QSVC
    # X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    # print(len(X))
    # print(len(y))

    # Split dataset into Training and Testing Data
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=32)

    # Define feature maps and kernel
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2)
    qkernel = FidelityQuantumKernel(feature_map=feature_map)

    # algorithm_globals.random_seed = 12345  # Already declared in the global variables
    # Define the Pegasos QSVC classifier
    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
    # training - fit classifier to data
    start = timer()
    pegasos_qsvc.fit(X, y)
    end = timer()
    train_time = end - start

    return pegasos_qsvc, train_time


# Callback function that draws a live plot when the .fit() method is called
def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    #plt.show()


# Predicting with any model
def predictModel(model, x):
    print('Prediction on model :', model)
    start = timer()
    predictions = model.predict(x)
    end = timer()
    predict_time = end - start
    return predictions, predict_time


# Apply SMOTE oversampling and Random Under Sampling
def apply_smote_under(X_pca, y_pca, s, k):
    over = SMOTE(sampling_strategy=s, k_neighbors=k)
    under = RandomUnderSampler(sampling_strategy=0.5)
    X, y = over.fit_resample(X_pca, y_pca)
    X, y = under.fit_resample(X, y)
    return X, y


# Save the qboost model for prediction
def saveModel(model, name):
    # -- Save the model using Pickle --
    # model_p = 'model.pkl'
    # pickle.dump(model, open(model_p, 'wb'))
    # -- Save the model using Joblib --
    model_file = name + '.pkl'
    joblib.dump(model, model_file)
    print('Model Saved :', model)


#  Load the model
def loadModel(model_file):
    # -- Loading Model from Pickle ----
    # modelQboost = pickle.load(open(model_file, 'rb'))
    # -- Loading Model from Joblib ----
    qboost = joblib.load(model_file)
    print(qboost)
    return qboost


# Reconstruction of Data for further usage
def reConstructData(predictions, x_test):
    # Convert the np array to dataframe
    df_predictions = pd.DataFrame(predictions, columns=['label'])

    # Concat dataframes horizontally
    result_data = pd.concat([x_test, df_predictions], axis=1)

    return result_data


# Generate a random alphanumeric string in Python
def random_alphanumeric_string(length):
    return ''.join(
        random.choices(
            string.ascii_letters + string.digits,
            k=length
        )
    )


# Create the primary key for insertion into the database
def createKey(res):
    key = []
    if len(res.shape) == 1:
        date_object = datetime.strptime(res[0].strip(), '%d-%b-%y').date()
        date_str = date_object.strftime("%y%m%d")
        time_object = datetime.strptime(res[1].strip(), '%H:%M:%S').time()
        time_str = time_object.strftime("%H%M%S")
        key = date_str.strip() + time_str.strip() + random_alphanumeric_string(5).strip()
    else:
        for data in res:
            date_object = datetime.strptime(data[0].strip(), '%d-%b-%y').date()
            date_str = date_object.strftime("%y%m%d")
            time_object = datetime.strptime(data[1].strip(), '%H:%M:%S').time()
            time_str = time_object.strftime("%H%M%S")
            keyval = date_str.strip() + time_str.strip() + random_alphanumeric_string(5).strip()
            key = np.append(key, keyval)
    # print(key)
    # exit(0)
    return key


# Delete rows from Predictions if needed
def delRow(data, num, col):
    return data[data[:, col] != num, :]


# Make appropriate changes in the data to update the table
def insertData(result_data):
    # The part below formats the data for insertion into DB
    # Convert Dataframe to array for insertion
    result_data = result_data.values

    # print(result_data)
    print(len(result_data.shape))

    # Create new key for the DB table
    res = createKey(result_data)

    # Insert a new first column with the created keys
    result_data = np.insert(result_data, 0, res, axis=1)

    # Delete rows from result where the results are NOT ATTACK
    # result_data = delRow(result_data, -1, (result_data.shape[1] - 1))  # TO DO: Analyse and remove this to permit
    # all data

    # Process the result_data for DB upload
    # Store the results to be uploaded in an array
    result_data[:, 3] = 'No Info'  # TO DO: Change this to include a summary of data that entered
    result_data[:, -2] = 'IoT'
    result_data[:, -1] = 'Attack State'  # TO DO: Change this to reflect all data from the dataset

    # Delete 5th (0, 1, 2, 3, 4<-) column(axis=1)  from the result_data np array
    result_data = np.delete(result_data, obj=4, axis=1)

    # result_data = np.array([str(data).strip() for data in result_data])

    print(result_data)

    # print("Predicted Value (x[" + str(randomtestnum) + "]):" + str(predictions[0]))
    # print("Test Value (y[" + str(randomtestnum) + "]):" + str(y_test[randomtestnum]))
    # print(testdata)
    # print(predictions)

    updateDB(result_data)

    print('DB Updated')


#  Insert results into DB
def updateDB(values):
    # Define the SQL queries here
    sql = "insert into scmobility.iotweather(attackid, attackdate, attacktime, severity, " \
          "categories, attacktype) values(%s, %s, %s, %s, %s, %s);"
    connect(sql, values)
    return 0


# Use arguments to run the program 0-->Classifier Type; 1-->Number of Components in PCA;
def main():
    if len(sys.argv) == 1:
        print('No arguments provided. Please provide one among qboost, vqc and pegasos. Exiting now.')
        exit(0)
    args = sys.argv[1:]
    print(args)

    # Number of components considered for the experiment (2nd argument while running program)
    num = int(args[1])

    # Load data into dataframe
    dataframe1 = loadData(path1)
    dataframe2 = loadData(path2)
    # Check
    print(dataframe1.head())
    print(dataframe2.head())

    # Make a copy of the dafaframe
    #df = dataframe

    # Merge the two dataframes
    df = pd.concat([dataframe1, dataframe2])

    # Clean data: Remove columns which have same numbers, categorical columns, since those are IP addresses and
    # one IP address leads to 32 features increasing the complexity of the dataset by manifolds
    df = dataClean(df)

    # Impute NaN and Infinity values by Median
    df = impute(df)  # used to calculate the data dimension

    # Separate Independent and Dependent Data
    X, y = create_xy(df)

    # Scaling and applying PCA
    X_pca = apply_pca(X, args[0], num)

    # Use QBoost to find out the best parameters for SMOTE and Random Undersampling
    # evaluate_over_under(X_pca, y_pca)
    # Commented since this step was already done and the values of s and k are 0,5 and 6

    # Commented the over/ undersampling since the dataset is balanced
    # SMOTE & undersampling values
    #s = 0.5  # Oversampling ratio
    #k = 6  # KNN for SMOTE Oversampling
    ## Undersampling is 0.5 by default, later can be changed for further experiments.
    #
    ## Apply SMOTE and Random Under Sampling to recreate a balanced dataset
    #X, y = apply_smote_under(X_pca, y, s, k)

    # Convert the pandas series into numpy array
    y_pca = y.values.ravel()

    # Declaring data dimension for qiskit
    # data_dimension = len(X.columns.values.tolist())
    data_dimension = X_pca.shape[1]
    print('Data Dimension :', data_dimension)

    # Check
    # print(Counter(y))

    # Declare model, and training time (remove warnings)
    model = ''
    train_time = ''

    # Split dataset into Training and Testing Data
    x_train, x_test, y_train, y_test = train_test_split(X_pca, y_pca, test_size=0.3, shuffle=True, random_state=32)

    if args[0] == 'qboost':
        # Training of QBoost dataset
        model, train_time = trainQboost(x_train, y_train)
    elif args[0] == 'vqc':
        # Training VQC - variational quantum classifier
        model, train_time = trainVQC(x_train, y_train, data_dimension)
    elif args[0] == 'pegasos':
        # Training PegaSOS QSVC - Needs data to be prepared in a different manner
        model, train_time = trainPQSVC(x_train, y_train, data_dimension)
    else:
        print('No argument mentioned. Exiting')
        exit(0)
     
    print('Training complete...')

    # Prediction results with training data
    predictions, predict_time = predictModel(model, x_train)
    reportResult(y_train, predictions, str(model)[1:6], 'training', train_time, predict_time)

    # If carrying out the prediction later need to save the model and change the program to include the line below
    # Load model
    # model = loadModel('qboost.pkl')  # TO DO: Change the code to include the prediction part separately

    # Prediction results with test data
    predictions, predict_time = predictModel(model, x_test)
    reportResult(y_test, predictions, args[0], num, 'test', train_time, predict_time)

    # Reconstruct data
    result_data = reConstructData(predictions, x_test)

    # Insert data into tables
    insertData(result_data)

    print('End Program !!!')


if __name__ == "__main__":
    main()
