import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing

def make_predictions(savefile):
    #Data Processing
    #Reading Data
    df = pd.read_csv(f"Stock Predictor 2.0/{savefile}.csv",index_col="Index")
    file = open("Stock Predictor 2.0\colList.txt", "r")
    data = file.readline()
    file.close()
    col = list(data.split("#"))
    
    #Dropping rows to match model's training data
    X = df[col].copy()
    nulldf = df[X.isnull().any(axis=1)]
    df = df[X.notna().all(axis=1)]
    X.dropna(inplace = True)
    
    print(df)

    #Encoding
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load('Stock Predictor 2.0\classes.npy')
    X["sector"] = le.transform(X["sector"])

    #Model
    loaded_model = pickle.load(open("Stock Predictor 2.0/finalized_model.sav", 'rb'))
    pred = loaded_model.predict(X)
    symbols = df[pred]
    

    #Output
    if not nulldf.empty:
        print(f"{len(nulldf)} Stock Information has Null Values:")
        for row in nulldf.values:
            print(f"({row[0]} | {row[1]})", end = " ")
    if not symbols.empty:
        print("\n\n--- Buy Tickers ---")
        print(symbols)
        for i in symbols.index:
            sym = symbols.loc[i, "symbol"]
            price = symbols.loc[i, "Stock Price"]
            tlow = symbols.loc[i, "targetLowPrice"]
            tmed = symbols.loc[i, "targetMedianPrice"]
            fair = symbols.loc[i, "priceFairValue"]
            print("| Symbol | Price | Target Low | Target Median | Fair Value |")
            print(f"|{sym:^8}|{price:^7.2f}|{tlow:^12.2f}|{tmed:^15.2f}|{fair:^12.2f}|")
        print(symbols[["symbol","Stock Price","targetLowPrice","targetMedianPrice","priceFairValue"]])
    else:
        print(f"\n\nNo buy predictions among {len(X)} stocks!")
    
def make_predictions_df(df):
    #Data Processing
    #Reading Data
    file = open("Stock Predictor 2.0\colList.txt", "r")
    data = file.readline()
    file.close()
    col = list(data.split("#"))
    
    #Dropping rows to match model's training data
    X = df[col].copy()
    nulldf = df[X.isnull().any(axis=1)]
    df = df[X.notna().all(axis=1)]
    X.dropna(inplace = True)
    
    print(df)

    #Encoding
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load('Stock Predictor 2.0\classes.npy')
    X["sector"] = le.transform(X["sector"])

    #Model
    loaded_model = pickle.load(open("Stock Predictor 2.0/finalized_model.sav", 'rb'))
    pred = loaded_model.predict(X)
    symbols = df[pred]
    

    #Output
    text = ""
    if not nulldf.empty:
        print(f"{len(nulldf)} Stock Information has Null Values:")
        for row in nulldf.values:
            print(f"({row[0]} | {row[1]})", end = " ")
    if not symbols.empty:
        print("\n\n--- Buy Tickers ---")
        for i in symbols.index:
            sym = symbols.loc[i, "symbol"]
            price = symbols.loc[i, "Stock Price"]
            tlow = symbols.loc[i, "targetLowPrice"]
            tmed = symbols.loc[i, "targetMedianPrice"]
            fair = symbols.loc[i, "priceFairValue"]
            t = (f"|{sym:^10}|{price:^7.2f}|{tlow:^17.2f}|{tmed:^25.2f}|{fair:^16.2f}|")
            text += "\n" + t
        return("| Symbol | Price | Target Low | Target Median | Fair Value |"+text)
    else:
        return(f"\n\nNo buy predictions among {len(X)} stocks!")


make_predictions("test")