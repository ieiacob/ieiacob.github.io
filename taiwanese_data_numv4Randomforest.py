# -*- coding: utf-8 -*-
"""Taiwanese DATA

v1: We randomly select NA attribute then create and train NM models.
Then we select the top models, we train again their outputs to produce the correct decision
NO feature adjustments!
"""
#%%******************************************** imports ************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import Precision
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from tensorflow import random as tfrandom
from sklearn.ensemble import RandomForestClassifier
import random
from collections import Counter
import sys
from tensorflow.keras.utils import plot_model
#%%============================ the sub-models ========================================================
def build_model(in_dim):
    #% Define the neural network model
    tfrandom.set_seed(2024)
    model = Sequential(name="Input")
    
    # Input layer and first hidden layer with 32 neurons and LeakyReLU activation
    model.add(Dense(8, input_dim=in_dim, activation='LeakyReLU', name="Hidden_1"))
    
    # Second hidden layer with 16 neurons
    model.add(Dense(4, activation='LeakyReLU', name="Hidden_2"))
    
    # Output layer with 1 neuron and sigmoid/tanh activation for binary classification
    model.add(Dense(1, activation='tanh', name="Out"))
    
    # Compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision()])
    
    return model


def build_outmodel(in_dim):
    #% Define the neural network model
    tfrandom.set_seed(2024)
    model = Sequential()
    
    # Input layer and first hidden layer with 32 neurons and ReLU activation
    model.add(Dense(8, input_dim=in_dim, activation='tanh'))
    
    # Second hidden layer with 16 neurons
    model.add(Dense(4, activation='tanh'))
    
    # Output layer with 1 neuron and sigmoid/tanh activation for binary classification
    model.add(Dense(1, activation='tanh'))
    
    # Compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision()])
    
    return model

def build_fullmodel(in_dim):
    #% Define the neural network model
    tfrandom.set_seed(2024)
    model = Sequential()
    
    # Input layer and first hidden layer with 32 neurons and ReLU activation
    model.add(Dense(32, input_dim=in_dim, activation='LeakyReLU', name="Hidden_1"))
    
    # Second hidden layer with 16 neurons
    model.add(Dense(4, activation='LeakyReLU', name="Hidden_2"))
    
    # Output layer with 1 neuron and sigmoid/tanh activation for binary classification
    model.add(Dense(1, activation='tanh', name="Out"))
    
    # Compile the model
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision()])
    
    return model



#%%========================================== the model =======================================================================
def top_majority(ys, accs, precs, y_test, nom, byprec = True):
    if byprec:
        nom = np.min((nom, sum(np.array(precs) > 0)))
        selmodels = sorted(range(len(precs)), key=lambda k: precs[k], reverse = True)[:nom]
    else:
        nom = np.min((nom, sum(np.array(accs) > 0)))
        selmodels = sorted(range(len(accs)), key=lambda k: accs[k], reverse = True)[:nom]

    selys = ys[:,selmodels]
    ys_binary = (selys >= 0.5).astype(int)
    ymaj_pred = np.sum(ys_binary, axis = 1)
    y_pred_binary = (ymaj_pred > (int)(nom / 2)).astype(int)

    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    TPR = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
    TNR = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    return selmodels, accuracy,precision,TNR,TPR,conf_matrix


def trained_topmodels(accs, trainys, y_train, ys, y_test, nom):
    nom = np.min((nom, sum(np.array(precs) > 0)))
    selmodels = sorted(range(len(precs)), key=lambda k: precs[k], reverse = True)[:nom]
    
    #scaler = StandardScaler()
    #y_out_results = scaler.fit_transform(trainys[:,selmodels])
    #selys = scaler.fit_transform(ys[:,selmodels])
    y_out_results = trainys[:,selmodels]
    selys = ys[:,selmodels]
    #% Define the neural network model
    model = build_outmodel(y_out_results.shape[1])
    #% Train the model
    y_out_results_binary = (y_out_results >= 0.5).astype(int)
    selys_binary = (selys >= 0.5).astype(int)
    model.fit(y_out_results_binary, y_train, epochs=400, batch_size=32, validation_data=(selys_binary, y_test), verbose = 0)
   
    y_pred = model.predict(selys)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    # Calculate accuracy and precision
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    TPR = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
    TNR = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    return accuracy,precision,TNR,TPR,conf_matrix
    
def full_model(X_train, y_train, X_test, y_test):
    model = model = build_fullmodel(X_train.shape[1])

    #% Train the model
    model.fit(X_train, y_train, epochs=400, batch_size=32, validation_data=(X_test, y_test), verbose = 0)
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)
    
    conf_matrix = confusion_matrix(y_test, y_pred_binary)

    #% Evaluate the model
    loss, accuracy, precision = model.evaluate(X_test, y_test)
    TPR = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0])
    TNR = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1])
    return accuracy,precision,TNR,TPR,conf_matrix

def random_forest_model(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(n_estimators=100, random_state=2024, max_depth=10)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    TPR = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
    TNR = conf_matrix[0,0] / (conf_matrix[0,0] + conf_matrix[0,1]) if (conf_matrix[0,0] + conf_matrix[0,1]) > 0 else 0
    return accuracy, precision, TNR, TPR, conf_matrix

#%%-------------------------------------------- top full model features ---------------------------
topfullmodelfeatures = [' ROA(C) before interest and depreciation before interest',
 ' Continuous interest rate (after tax)',
 ' Net Value Per Share (C)',
 ' Persistent EPS in the Last Four Seasons',
 ' Cash Flow Per Share',
 ' Operating Profit Per Share (Yuan ¥)',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Total Asset Growth Rate',
 ' Total Asset Return Growth Rate Ratio',
 ' Current Ratio',
 ' Quick Ratio',
 ' Total debt/Total net worth',
 ' Net worth/Assets',
 ' Borrowing dependency',
 ' Net profit before tax/Paid-in capital',
 ' Net Worth Turnover Rate (times)',
 ' Current Assets/Total Assets',
 ' Current Liability to Assets',
 ' Operating Funds to Liability',
 ' Inventory/Working Capital',
 ' Current Liabilities/Liability',
 ' Total income/Total expense',
 ' Total expense/Assets',
 ' Current Asset Turnover Rate',
 ' Working capitcal Turnover Rate',
 ' Equity to Long-term Liability',
 ' Liability-Assets Flag',
 ' Liability to Equity',
 ' Net Income Flag',
 ' Equity to Liability']

topfullmodelfeatures = [' Net Income to Total Assets',
 ' Total debt/Total net worth',
 ' Continuous interest rate (after tax)',
 ' Liability-Assets Flag',
 ' Net Value Per Share (C)',
 ' Working Capital/Equity',
 ' Cash Flow to Total Assets',
 ' Inventory/Working Capital',
 ' ROA(B) before interest and depreciation after tax',
 ' Realized Sales Gross Margin',
 ' Accounts Receivable Turnover',
 ' Cash Flow Per Share',
 ' Interest Coverage Ratio (Interest expense to EBIT)',
 ' Non-industry income and expenditure/revenue',
 ' Total Asset Turnover',
 ' Cash Flow to Sales',
 ' Continuous Net Profit Growth Rate',
 ' Current Liabilities/Equity',
 ' Current Liability to Assets',
 ' Current Liability to Liability',
 ' Debt ratio %',
 ' Equity to Liability',
 ' Liability to Equity',
 ' Net profit before tax/Paid-in capital',
 ' Operating Funds to Liability',
 ' Pre-tax net Interest Rate',
 ' ROA(A) before interest and % after tax',
 ' Research and development expense rate',
 ' Retained Earnings to Total Assets',
 ' Tax rate (A)',
 ' Total expense/Assets',
 ' Allocation rate per person',
 ' Borrowing dependency',
 ' CFO to Assets',
 ' Cash Turnover Rate',
 ' Cash/Current Liability',
 ' Current Assets/Total Assets',
 ' Interest Expense Ratio',
 ' Operating Expense Rate',
 ' Operating Gross Margin',
 ' Realized Sales Gross Profit Growth Rate',
 ' Working Capital to Total Assets',
 ' Cash Flow to Equity',
 ' Cash Reinvestment %',
 ' Cash/Total Assets',
 ' Current Liabilities/Liability',
 ' Current Liability to Current Assets',
 ' Current Liability to Equity',
 ' Fixed Assets to Assets',
 ' Gross Profit to Sales',
 ' Net Income Flag',
 ' Net worth/Assets',
 ' No-credit Interval',
 ' Operating Profit Growth Rate',
 ' Operating Profit Rate',
 ' Operating profit/Paid-in capital',
 ' Quick Assets/Total Assets',
 ' ROA(C) before interest and depreciation before interest',
 ' Working capitcal Turnover Rate',
 ' Contingent liabilities/Net worth',
 ' Degree of Financial Leverage (DFL)',
 ' Interest-bearing debt interest rate',
 ' Long-term fund suitability ratio (A)',
 ' Net Value Per Share (A)',
 ' Net Value Per Share (B)',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Quick Ratio',
 ' Total Asset Return Growth Rate Ratio',
 ' Total income/Total expense']

topfullmodelfeatures = [' Debt ratio %',
 ' Operating Expense Rate',
 ' Cash Flow to Liability',
 ' Inventory and accounts receivable/Net value',
 ' Net Value Per Share (C)',
 ' Net worth/Assets',
 ' Total assets to GNP price',
 ' Working Capital/Equity',
 ' Allocation rate per person',
 ' Cash Flow to Total Assets',
 ' Cash Turnover Rate',
 ' Continuous Net Profit Growth Rate',
 ' Current Assets/Total Assets',
 ' Current Liability to Current Assets',
 ' Liability to Equity',
 ' Net profit before tax/Paid-in capital',
 ' Non-industry income and expenditure/revenue',
 ' After-tax net Interest Rate',
 ' Borrowing dependency',
 ' CFO to Assets',
 ' Cash Flow to Sales',
 ' Cash Reinvestment %',
 ' Current Liability to Assets',
 ' Degree of Financial Leverage (DFL)',
 ' Interest Coverage Ratio (Interest expense to EBIT)',
 ' Net Income to Total Assets',
 ' Net Worth Turnover Rate (times)',
 ' Operating Gross Margin',
 ' ROA(A) before interest and % after tax',
 ' ROA(B) before interest and depreciation after tax',
 ' ROA(C) before interest and depreciation before interest',
 ' Total debt/Total net worth',
 ' Total expense/Assets',
 ' Cash Flow Per Share',
 ' Cash/Total Assets',
 ' Continuous interest rate (after tax)',
 ' Current Ratio',
 ' Fixed Assets to Assets',
 ' Gross Profit to Sales',
 ' Long-term Liability to Current Assets',
 ' Net Income Flag',
 ' Operating Profit Growth Rate',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Pre-tax net Interest Rate',
 ' Quick Asset Turnover Rate',
 ' Regular Net Profit Growth Rate',
 ' Retained Earnings to Total Assets',
 ' Tax rate (A)',
 ' Total income/Total expense']


topfullmodelfeatures = [' Net worth/Assets',
 ' Allocation rate per person',
 ' Total debt/Total net worth',
 ' Working Capital/Equity',
 ' Borrowing dependency',
 ' Cash Flow to Liability',
 ' Debt ratio %',
 ' Inventory and accounts receivable/Net value',
 ' ROA(B) before interest and depreciation after tax',
 ' Non-industry income and expenditure/revenue',
 ' Tax rate (A)',
 ' Cash Turnover Rate',
 ' Continuous Net Profit Growth Rate',
 ' Current Liability to Current Assets',
 ' Degree of Financial Leverage (DFL)',
 ' Operating Expense Rate',
 ' Operating Profit Rate',
 ' Pre-tax net Interest Rate',
 ' After-tax net Interest Rate',
 ' Cash flow rate',
 ' Cash/Current Liability',
 ' Continuous interest rate (after tax)',
 ' Current Liability to Assets',
 ' Persistent EPS in the Last Four Seasons',
 ' Realized Sales Gross Profit Growth Rate',
 ' Total assets to GNP price',
 ' Total expense/Assets',
 ' Current Assets/Total Assets',
 ' Current Liabilities/Liability',
 ' Gross Profit to Sales',
 ' Interest Coverage Ratio (Interest expense to EBIT)',
 ' Liability to Equity',
 ' Net Income to Total Assets',
 ' Net Value Per Share (C)',
 ' Net Worth Turnover Rate (times)',
 ' Operating Gross Margin',
 ' Operating profit/Paid-in capital',
 ' Quick Assets/Total Assets',
 ' ROA(A) before interest and % after tax',
 ' ROA(C) before interest and depreciation before interest',
 ' Retained Earnings to Total Assets',
 ' Total Asset Turnover',
 ' Working Capital to Total Assets',
 ' CFO to Assets',
 ' Cash Flow Per Share',
 ' Cash Flow to Sales',
 ' Cash Flow to Total Assets',
 ' Cash Reinvestment %',
 ' Cash/Total Assets',
 ' Contingent liabilities/Net worth',
 ' Current Liabilities/Equity',
 ' Equity to Long-term Liability',
 ' Fixed Assets to Assets',
 ' Interest-bearing debt interest rate',
 ' Inventory/Current Liability',
 ' Long-term Liability to Current Assets',
 ' Net Value Growth Rate',
 ' Net profit before tax/Paid-in capital',
 ' Operating Profit Per Share (Yuan ¥)',
 ' Per Share Net profit before tax (Yuan ¥)',
 ' Quick Asset Turnover Rate',
 ' Revenue Per Share (Yuan ¥)',
 ' Total Asset Return Growth Rate Ratio']

allexp_top_models_features = []

#%%----------------------------------------------- Experiment ID -------------------------------------------
SEEDS = [12, 335, 567, 847, 3026, 20973, 45, 311123, 43899, 9021] #10 experiments
CLIPPING1 = [0, 1]
CLIPPING2 = [0.1, 0.9]
CLIPPING3 = [0.25, 0.75]

EXPCLIPPING = CLIPPING1
EXPID = 0 #change this from 0 to 9 for each experiment run    

#reset features set
if EXPID == 0:
    allexp_top_models_features = []



for EXPID in range(len(SEEDS)):
    
    #%% Load the data, cleaning
    
    TaiwaneseBankruptcyPred_df = pd.read_csv('../data/taiwanesebankruptcyprediction/data.csv')
    
    
    cols = TaiwaneseBankruptcyPred_df.columns
    cols = np.concatenate((cols[1:], [cols[0]]))
    TaiwaneseBankruptcyPred_df = TaiwaneseBankruptcyPred_df[cols]
    #select only about twice zeros than ones, to balance the data
    ones = np.array(TaiwaneseBankruptcyPred_df[[cols[-1]]]==1)
    indices = np.where(~ones)[0]
    noones = (int) (np.sum(ones))
    #ones[indices[random.sample(list(range(len(indices))), 2*noones)]] = True
    random.seed(SEEDS[EXPID])
    ones[indices[random.sample(list(range(len(indices))), 330)]] = True
    TaiwaneseBankruptcyPred_df = TaiwaneseBankruptcyPred_df[ones]
    
    TaiwaneseBankruptcyPred_df[[cols[-1]]] = 1 - TaiwaneseBankruptcyPred_df[[cols[-1]]] #TaiwaneseBankruptcyPred_df[[cols[-1]]].replace(0, -1)
    
    #random.seed(33)    #(54321,111) -> 105; (4321, 33) -> 106
    Source_df = TaiwaneseBankruptcyPred_df.reset_index(drop=True) #TaiwaneseBankruptcyPred_df.sample(frac=1).reset_index(drop=True)
    
    #%%--------------------------------------- some boxplots, clipping ------------------------------------
    attribute = ' Research and development expense rate'
    attribute = ' Operating Expense Rate'
    
#    fig, ax = plt.subplots()
    
    # Creating plot
    data1 = Source_df.loc[:,[attribute]]
    
    #quantiles = Source_df.quantile(EXPCLIPPING, axis = 0)
    ##clip data
    #lower_limits = np.array(quantiles.iloc[[0],:])
    #upper_limits = np.array(quantiles.iloc[[1],:])
    #Source_df <- Source_df.clip(lower_limits, upper_limits, axis = 1)
    if False:  #turn pre-processing ON/OFF
        for c in Source_df.columns[:-1]:
            median = Source_df[c].median()
            std = Source_df[c].std()
            #outliers = (Source_df[c] - median).abs() > 1.75*std
            outliers = Source_df[c] > 1
            Source_df.loc[outliers, c] = 1
            #Source_df[c].fillna(median, inplace=True)
    
    # Creating plot
    data2 = Source_df.loc[:,[attribute]]
#    ax.boxplot([np.array(data1).flatten(),np.array(data2).flatten()],labels=['before','after'])
    # Set title and labels
#    ax.set_title('Variable before and after clipping')
#    ax.set_xlabel(attribute)
#    ax.set_ylabel('Values')
    
#    plt.show()
    
    #scaler = StandardScaler()
    #data = scaler.fit_transform(Source_df)[:,[28]]
    #bp = ax.boxplot(data)
    #ax = fig.add_axes([0, 0, 1, 1])
    # show plot
    #plt.show()
    
    #German_df.info()
    #%%----------------------------------------- parameters -----------------------------------
    USETOPFEATURES = False
    
    
    TNM = 7           #the top number of models
    
    cols = Source_df.columns
    
    TOTNC = len(cols)-1             #total number of columns
    predictedIDX = TOTNC
    
    
    if USETOPFEATURES:
        TOTNC = len(topfullmodelfeatures)
    
    
    NROWS = len(Source_df)        #number of rows
    
    PTEST = 0.3                  #fraction of total rows for testing
    NROWSTEST = (int) (NROWS * PTEST)
    
    NA = 10#(int)(TOTNC / TNM) + 1            #number of variables
    NM = 25#(int)(TOTNC * 2 / NA)            #number of models
    
    
    EPOCHS = 250
    
    NTESTNEG = 53
    
    
    #%% am splitting the Data , that is train and test Data**"""
    
    predicted = cols[predictedIDX]
    # Encode the 'Risk Level' column into binary values
    #label_encoder = LabelEncoder()
    #Source_df[predicted] = label_encoder.fit_transform(Source_df[predicted])  # 'Good' -> 1, 'Bad' -> 0
    #Source_df[predicted] = 1 - Source_df[predicted]
    
    
    #% Split the dataset into training and testing sets (80% train, 20% test)
    X = Source_df.drop(columns=cols[predictedIDX])
    if USETOPFEATURES:
        X = X[topfullmodelfeatures]
    cols = X.columns
    if USETOPFEATURES:
        TOTNC = len(topfullmodelfeatures)
    y = Source_df[predicted]
    
    neg = np.array(y==0)
    indicesneg = np.where(neg)[0]
    indicespos = np.where(~neg)[0]
    #ones[indices[random.sample(list(range(len(indices))), 2*noones)]] = True
    random.seed(4321)
    testidx = np.concatenate([np.array(random.sample(list(indicespos), NROWSTEST-NTESTNEG)), 
                             np.array(random.sample(list(indicesneg), NTESTNEG))])
    
    #X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=PTEST, shuffle=False, stratify=None, random_state=501)
    X_test1 = X[X.index.isin(testidx)]
    y_test = y[y.index.isin(testidx)]
    X_train1 = X[~X.index.isin(testidx)]
    y_train = y[~y.index.isin(testidx)]
    
    scaler = StandardScaler()
    print(sum(y_test))
    
    #%%----------------------------------------- train the models ---------------------------------------------
    #ensure replication
    #random.seed(2024)
    random.seed(SEEDS[EXPID])
    
    y_out_results = []
    y_pred_results = []
    accs = []
    precs = []
    selected_cols = []
    
    for i in range(0, NM):
        print(f'MODEL #{i+1}/{NM}')
        #select random columns to drop
        selcol = random.sample(range(0,TOTNC), TOTNC - NA)
        
        X_train0 = X_train1.drop(columns=cols[selcol])
        X_test0 = X_test1.drop(columns=cols[selcol])
        print(X_test0.columns)
        selected_cols.append(X_test0.columns)
         
    
        X_train0.shape, X_test0.shape, y_train.shape, y_test.shape
        
        #%*************************************** scale data ****************************************************
        
        # Standardize numerical features
        X_train = scaler.fit_transform(X_train0)
        X_test = scaler.fit_transform(X_test0)
        
        
        #% Define the neural network model
        model = build_model(X_train.shape[1])
            
        #% Train the model
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(X_test, y_test), verbose = 0)
        
        #% Evaluate the model
        loss, accuracy, precision = model.evaluate(X_test, y_test)
        print(f'Test Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        accs.append(accuracy)
        precs.append(precision)
        
        y_out = model.predict(X_train)
        y_out_results.append(np.concatenate(y_out).reshape(-1))
        
        y_pred = model.predict(X_test)
        y_pred_results.append(np.concatenate(y_pred).reshape(-1))
    
    
    #selmodels = range(0,NM-1)
    #selmodels = random.sample(range(0,NM-1), TNM)
    #selmodels = sorted(range(len(accs)), key=lambda k: accs[k], reverse = True)[:TNM]
    
    y_out_results1 = np.concatenate(y_out_results).reshape((NM,X_train1.shape[0])).T
    y_pred_results1 = np.concatenate(y_pred_results).reshape((NM,X_test1.shape[0])).T
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision()])
    
    #y_pred_binary = (y_pred_count > (int)(NM / 2)).astype(int)
    #y_pred_results_sorted = sorted(y_pred_results, key = lambda x: x[0], reverse = True)
    #y_pred_count2 = np.stack(y_pred_results_sorted[0:TNM], axis = 0)[:,2:].sum(axis = 0)
    #y_pred_binary2 = (y_pred_count2 > (int)(TNM / 2)).astype(int)
    
    #%%========================================= Clipping and Exp ID =====================================
    orig_stdout = sys.stdout
    f = open('results.txt', 'a')
    sys.stdout = f
    print(f'************ Experiment ID: {EXPID}')
    
    #%%================================ top max majority model ===========================================
    selmodels, accuracy, precision, TNR, TPR, conf_matrix = top_majority(y_pred_results1, accs, precs, y_test, 7, byprec = False)
    conf_matrix_df = pd.DataFrame(conf_matrix,
                                  index=['Actual: Bad', 'Actual: Good'],
                                  columns=['Predicted: Bad', 'Predicted: Good'])
    
    print("Majority Model")
    print("============================")
    print("Confusion Matrix:")
    print(conf_matrix_df)
    
    # Display accuracy and precision
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'TNR: {TNR}')
    print(f'TPR: {TPR}')
    sys.stdout = orig_stdout
    f.close()
    #%%======================================= Top Attributes count ======================================
    top_models_features_occurrences = np.concatenate([selected_cols[i] for i in selmodels])
    allexp_top_models_features.extend(top_models_features_occurrences)
    
    top_models_features_count = Counter(top_models_features_occurrences)
    allexp_top_models_features_count = Counter(allexp_top_models_features)

#%%==============================================sort by attribute
sorted(top_models_features_count.items())
sorted(allexp_top_models_features_count.items())

#sort by occurrences, attribute
#sorted(allexp_top_models_features_count.items(), key=lambda item: (-item[1], item[0]), reverse=False)
#sorted(dict(tai_top_features).items(), key=lambda item: (-item[1], item[0]), reverse=False)

#%%=============================== trained top models outputs =================================
# accuracy, precision, TNR, TPR, conf_matrix = trained_topmodels(accs, scaler.fit_transform(y_out_results1), y_train, scaler.fit_transform(y_pred_results1), y_test, NM)
# conf_matrix_df = pd.DataFrame(conf_matrix,
#                               index=['Actual: Bad', 'Actual: Good'],
#                               columns=['Predicted: Bad', 'Predicted: Good'])

# print("Trained Majority Model")
# print("============================")
# print("Confusion Matrix:")
# print(conf_matrix_df)

# # Display accuracy and precision
# print(f'Accuracy: {accuracy}')
# print(f'Precision: {precision}')
# print(f'TNR: {TNR}')
# print(f'TPR: {TPR}')
#%%================================== full model ===========================================================
#top model features
USETOPFEATURES = False

full_model_features = topfullmodelfeatures
# Standardize numerical features
#X_trainfull = scaler.fit_transform(X_train1[full_model_features])
#X_testfull = scaler.fit_transform(X_test1[full_model_features])
#X_trainfull = scaler.fit_transform(X_train1)
#X_testfull = scaler.fit_transform(X_test1)

X_trainfull = X_train1
X_testfull = X_test1

if USETOPFEATURES:
    X_trainfull = X_train1[full_model_features]
    X_testfull = X_test1[full_model_features]


accuracy, precision, TNR, TPR, conf_matrix = full_model(X_trainfull, y_train, X_testfull, y_test)
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Actual: Bad', 'Actual: Good'],
                              columns=['Predicted: Bad', 'Predicted: Good'])

print("Full Model")
print("============================")
print("Confusion Matrix:")
print(conf_matrix_df)

# Display accuracy and precision
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'TNR: {TNR}')
print(f'TPR: {TPR}')

#%%============================Random forest=================================
# Standardize numerical features
#X_train = scaler.fit_transform(X_train1)
#X_test = scaler.fit_transform(X_test1)

X_train = X_train1
X_test = X_test1

if USETOPFEATURES:
    X_train = X_train1[full_model_features]
    X_test = X_test1[full_model_features]

accuracy, precision, TNR, TPR, conf_matrix = random_forest_model(X_train, y_train, X_test, y_test)
print("Random Forest Model")
print("============================")
print("Confusion Matrix:")
conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Actual: Bad', 'Actual: Good'],
                              columns=['Predicted: Bad', 'Predicted: Good'])
print(conf_matrix_df)
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'TNR: {TNR}')
print(f'TPR: {TPR}')


#%%========================================================================