import pickle
import numpy as np
import pandas as pd


#Models
from sklearn.ensemble import RandomForestClassifier,StackingClassifier
from xgboost import XGBClassifier 
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,cross_validate
from sklearn.metrics import precision_score,accuracy_score,recall_score
from sklearn.model_selection import KFold 

#Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from scipy import stats

#Checks classification scores
def classification_score(X,y,z,clf,cv=3,p=False):
    kf = KFold(n_splits=cv, shuffle = True, random_state=42) 
    train_acc_score = []
    train_pre_score = []
    train_rec_score = []
    test_acc_score = []
    test_pre_score = []
    test_rec_score = []
    stock_returns = []
    market_returns = []
    negative =[]
    predictions = []
    for train_index , test_index in (kf.split(X)):
        X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]
        y_train , y_test = y.iloc[train_index] , y.iloc[test_index]
        z_train , z_test = z[train_index] , z[test_index]

        #Train Scores
        clf.fit(X_train,y_train)
        train_pred_values = clf.predict(X_train)
        if sum(train_pred_values) == 0:
            print("No train predictions!")
            return()
        train_acc = accuracy_score(y_train, train_pred_values)
        train_pre = precision_score(y_train, train_pred_values, zero_division = 0)
        train_rec = recall_score(y_train, train_pred_values, zero_division = 0)
        train_acc_score.append(train_acc)
        train_pre_score.append(train_pre)
        train_rec_score.append(train_rec)
        
        #Test Scores
        pred_values = clf.predict(X_test)
        if sum(pred_values) == 0:
            print("No predictions!")
            return()
        predictions.append(sum(pred_values))
        test_acc = accuracy_score(y_test, pred_values)
        test_pre = precision_score(y_test, pred_values, zero_division = 0)
        test_rec = recall_score(y_test, pred_values, zero_division = 0) 
        [negative.append(val) for val in z_test[pred_values == 1, 0] if val<0]

        # Whenever a stock is predicted to outperform (pred_values = 1), we 'buy' that stock
        # and simultaneously `buy` the index for comparison.
        try:
            stock_returns.append(sum(z_test[pred_values == 1, 0])/sum(pred_values))
            market_returns.append(sum(z_test[pred_values == 1, 1])/sum(pred_values))
        except:
            stock_returns.append(0)
            market_returns.append(0)
        test_acc_score.append(test_acc)
        test_pre_score.append(test_pre)
        test_rec_score.append(test_rec)

    if p:
        print("--- Train Avg ---")
        print(f"Train Accuracy: {sum(train_acc_score)/cv*100: .2f}%")
        print(f"Train Precision: {sum(train_pre_score)/cv*100: .2f}%")
        print(f"Train Recall: {sum(train_rec_score)/cv*100: .2f}%")
        print("--- Test Avg ---")
        print(f"Test Accuracy: {sum(test_acc_score)/cv*100: .2f}%")
        print(f"Test Precision: {sum(test_pre_score)/cv*100: .2f}%")
        print(f"Test Recall: {sum(test_rec_score)/cv*100: .2f}%")

        total_outperformance = sum(stock_returns)/cv - sum(market_returns)/cv

        print("\n Stock prediction performance report \n", "=" * 40)
        print(f"Total Trades:", sum(predictions))
        print(f"Positive Trades %: {100-len(negative)/sum(predictions)*100:.2f}")
        print(f"Negative Trades %: {len(negative)/sum(predictions)*100:.2f}")
        #print(total_outperformance)
        print(f"Stock price Variance: {np.var(stock_returns): .5f}")
        print(f"Average return for stock predictions: {sum(stock_returns)/cv: .1f} %")
        print(f"Average market return in the same period: {sum(market_returns)/cv: .1f}% ")
        print(f"Compared to the index, our strategy earns {total_outperformance: .1f} percentage points more")
        return()
    
    return (sum(train_acc_score)/cv*100,sum(train_pre_score)/cv*100,sum(train_rec_score)/cv*100, sum(test_acc_score)/cv*100,
            sum(test_pre_score)/cv*100,sum(test_rec_score)/cv*100)

#Read data
df = pd.read_csv(f"Stock Predictor 2.0\FMP_YF_Data_US2.csv",index_col="Index")

#If column has more than 10% null drop
df = df.loc[:, df.isnull().mean() < 10/100]
df.dropna(axis = 0, how = 'any', inplace = True)
describtion = df.describe()
df.drop(["reportedCurrency","period"],axis=1,inplace=True)
df.reset_index(inplace=True)

### Data Cleaning
#Remove Outliers
lower_quartile = (df["Stock Change"]-df["Market Change"]).describe()[4]
higher_quartile = (df["Stock Change"]-df["Market Change"]).describe()[6]
quartile_range = higher_quartile - lower_quartile
lower_outlier = lower_quartile - 1.5*quartile_range
higher_quartile = higher_quartile + 1.5*quartile_range
df = df[(df["Stock Change"]-df["Market Change"]) <= higher_quartile]
df = df[(df["Stock Change"]-df["Market Change"]) >= lower_outlier]

#Plot Graph 
sns.histplot(data=(df["Stock Change"]-df["Market Change"]))
#plt.show()

#Stratified Sampling
buy = df[df['Buy']==1]
sell = df[df['Buy']==0]
df_new = sell.sample(n=round(len(buy)*2.5), random_state=42).copy()
df_new = pd.concat([buy,df_new],axis=0)
#df_new = df #Remove Stratified Sampling

#Initializing x, y and z
drop_col = ["date", "Index", "symbol", "name", "Buy", "Stock Change", "Market Change", "date"]
X = df_new.drop(columns = drop_col)
y = df_new['Buy']
z = np.array(df_new[['Stock Change','Market Change','symbol']])

#Encoding sector
le = preprocessing.LabelEncoder()
le.fit(['Healthcare', 'Basic Materials', 'Consumer Defensive', 'Financial Services',
        'Technology', 'Industrials', 'Consumer Cyclical', 'Real Estate', 'Utilities',
        'Communication Services', 'Energy'])
X["sector"] = le.transform(X["sector"])
np.save('Stock Predictor 2.0\classes.npy', le.classes_)

#Scaling data
min_max_scaler = preprocessing.MinMaxScaler()
X_processed = pd.DataFrame(min_max_scaler.fit_transform(X))
X_processed.columns = X.columns
X = X_processed

#Feature Pruning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=150, min_samples_leaf=75, bootstrap=True, max_features=0.7, n_jobs=-1)
clf.fit(X_train,y_train)
rf_feature_df = pd.concat((pd.DataFrame(X_train.columns, columns = ['feature']), 
           pd.DataFrame(clf.feature_importances_, columns = ['rf importance'])), 
          axis = 1)

clf = XGBClassifier(booster='dart', eta=0.13, eval_metric='logloss', gamma=5, max_depth=3,min_child_weight=3, random_state=42, reg_alpha=15, reg_lambda=15, use_label_encoder=False)
clf.fit(X_train,y_train)
feature_df = pd.concat((rf_feature_df, 
           pd.DataFrame(clf.feature_importances_, columns = ['xgb importance'])), 
          axis = 1).sort_values(by='xgb importance', ascending = True)

drop_lst = []
for row in feature_df.values:
    if row [1] <= 0.001 and row[2] <= 0.001:
        drop_lst.append(row[0])
X = X.drop(columns = drop_lst)

#Saving current DataFrame columns
file = open("Stock Predictor 2.0\colList.txt", "w")
file.write("#".join(list(X.columns)))
file.close()

###Model Creation
rf = RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=150, min_samples_leaf=75, bootstrap=True, max_features=0.7, n_jobs=-1)

#GridSearch
'''
xgb = XGBClassifier(booster='gbtree', eta=0.2, eval_metric='error', gamma=0, max_depth=3, 
                      min_child_weight=1, random_state=42, reg_alpha=25, reg_lambda=10, use_label_encoder=False)

hyper_parameters = {'booster': ["gbtree","dart"], 'eta': [0.1,0.3,0.5], 'gamma': [0, 3, 5], 
 'max_depth': [4, 6, 8], 'min_child_weight': [1,3,5], 'reg_lambda': [1,3,5], 'reg_alpha': [0,1,2,3],
 'use_label_encoder':[False], 'eval_metric':['logloss','error'], 'random_state': [42]}

xgb_gs = RandomizedSearchCV( n_iter=500, estimator=xgb, param_distributions=hyper_parameters, cv=3, random_state=42, 
                            verbose=10,n_jobs=-1)
xgb_gs.fit(X, y)
print(xgb_gs.best_params_)
xgb = xgb_gs.best_estimator_
'''

xgb = XGBClassifier(booster='dart', eta=0.13, eval_metric='logloss', gamma=5, max_depth=3, 
                      min_child_weight=3, random_state=42, reg_alpha=15, reg_lambda=15, use_label_encoder=False)

mlp = MLPClassifier(hidden_layer_sizes=(75,75,10),activation="relu",random_state=42,max_iter=1000)

#Stacking Classifier
lst = [("rf",rf),('xgb',xgb),('mlp',mlp)]

level1 = LogisticRegression()
# define the stacking ensemble
model = StackingClassifier(estimators=lst, final_estimator=level1, 
                           cv=3,n_jobs=None)

#print("RF")
#classification_score(X, y, z, rf,3,True)
#print("XGB")
#classification_score(X, y, z, xgb,3,True)
#print("MLP")
#classification_score(X, y, z, mlp,3,True)
print("Stacked")
classification_score(X, y, z, model,3,True)
#print("Done")

filename = 'finalized_model.sav'
pickle.dump(model, open("Stock Predictor 2.0/"+filename, 'wb'))

