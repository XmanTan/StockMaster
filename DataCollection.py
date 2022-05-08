import time
import traceback
import pandas as pd
import fmpsdk as fmp
from tqdm import tqdm
import yfinance as yf
import datetime as dt

apikeys = ['5c1fef454bf87788f4adc17145b6171d','aa63fe105f496bf566f949a0f20d4634','bcd5ccbc3b5777d738225ab4f1f43f9f','7c184f15f96dc492454eb6ab23b80f19','78b0c6f2b704c742938e6e4ffc4a1215',
           '68f0ccc432776bb1e3c39cbaf0e73eee','d66f6c3bc5b58a106c36d77147318106','1ca2d99b7f976277a15b4c345d620306','977051c39391ebe7848bdc7e871c256c','49f72d8d25ce312160efffaf646d5833',
           '76e09ee4f32e9f52657ac75d3f714dfc','7af2c4419f4483aed674bff3a749eaf3','895b434e7fb7d002eec33f4b4412eba3','477213bee65f540b8eddfc5891e78d0a','86552b7423e0d322202384b449876eb5'] 
stockExchanges = ["NYSE","NASDAQ","AMEX"]

#US["NYSE","NASDAQ","AMEX"] Count:7041
#EU
#tickerList = [ticker["symbol"] for ticker in fmp.symbols_list(apikey=apikeys[0]) if ticker["exchangeShortName"] in stockExchanges and ticker['type'] == 'stock'] 
#file = open("tickerList.txt","w")
#file.write('/'.join(tickerList))
#file.close()

file = open("Stock Predictor 2.0/tickerList.txt","r")
tickerList = file.read()
tickerList = tickerList.split('/')
file.close()

file = open("Stock Predictor 2.0/failedTickerList.txt","r")
failList = file.read()
failList = failList.split('/')
file.close()

file = open("Stock Predictor 2.0/count.txt","r")
count1 = int(file.read())
file.close()

#tickerList = fmp.symbols_list(apikey=apikeys[0])
#tickerList = fmp.financial_statement_symbol_lists(apikey=apikeys[0]) #Count: 26993
#print(len(tickerList))

#Extract and process income statment
def get_income(key,ticker):
    dataLst = fmp.income_statement(apikey=key,symbol=ticker) #Count: 10,35
    #Append using list comprehension then convert to df #Sub 4s /5
    header = dataLst[0].keys()
    lst = [data.values() for data in dataLst]
    df = pd.DataFrame(lst,columns=header)
    df.drop(['link','finalLink','fillingDate','acceptedDate','period'],axis=1,inplace=True)
    return df
    
#Extract and process income statment growth
def get_income_growth(key,ticker):
    dataLst = fmp.income_statement_growth(key,ticker) #Count: 10,29
    #Append using list comprehension then convert to df
    header = dataLst[0].keys()
    lst = [data.values() for data in dataLst]
    df = pd.DataFrame(lst,columns=header)
    return df

#Extract and process cashflow statment growth
def get_cashflow_growth(key,ticker):
    dataLst = fmp.cash_flow_statement_growth(apikey=key,symbol=ticker) #Count: 10,33
    #Append using list comprehension then convert to df
    header = dataLst[0].keys()
    lst = [data.values() for data in dataLst]
    df = pd.DataFrame(lst,columns=header)
    df.drop(["growthDepreciationAndAmortization","growthNetIncome"],axis=1,inplace=True)
    return df

#Extract and process key metrics
def get_key_metrics(key,ticker):
    dataLst = fmp.key_metrics(apikey=key,symbol=ticker) #Count: 10,60
    #Append using list comprehension then convert to df
    header = dataLst[0].keys()
    lst = [data.values() for data in dataLst]
    df = pd.DataFrame(lst,columns=header)
    return df

#Extract and process financial ratios
def get_financial_ratios(key,ticker):
    dataLst = fmp.financial_ratios(apikey=key,symbol=ticker) #Count: 10,60
    #Append using list comprehension then convert to df
    header = dataLst[0].keys()
    lst = [data.values() for data in dataLst] # 3.861s /5
    df = pd.DataFrame(lst,columns=header)
    return df

#Extract and process historical ratings
def get_hist_ratings(key,ticker,datadf):
    dataLst = fmp.historical_rating(apikey=key,symbol=ticker) #Count: 10,60
    #Append then convert to df
    lst = []
    header = dataLst[0].keys()
    for data in dataLst:
        lst.append(data.values())
    df = pd.DataFrame(lst,columns=header)
    df.drop(['ratingRecommendation','ratingDetailsDCFRecommendation','ratingDetailsROERecommendation',
    'ratingDetailsROARecommendation','ratingDetailsDERecommendation','ratingDetailsPERecommendation',
    'ratingDetailsPBRecommendation'],axis=1,inplace=True)
    print(df.head())
    dates = datadf["date"].tolist()
    #print(df["date"].isin(dates))
    df = df[df["date"].isin(dates)]
    return df

#Run all the get functions
def get_all(index,ticker,predict):
    global count1
    incomedf = get_income(apikeys[index],ticker)
    count1 += 1
    incomegrowthdf = get_income_growth(apikeys[index],ticker)
    count1 += 1
    cashflowgrowthdf = get_cashflow_growth(apikeys[index],ticker)
    count1 += 1
    keymetricsdf =  get_key_metrics(apikeys[index],ticker)
    count1 += 1
    financialratiodf = get_financial_ratios(apikeys[index],ticker)
    count1 += 1
    #histratingsdf = get_hist_ratings(apikeys[index],ticker,incomegrowthdf)
    #Combining horizontally in tempdf
    tempdf = pd.concat([incomedf, incomegrowthdf,cashflowgrowthdf,keymetricsdf,financialratiodf], axis=1, join='inner')
    #Cleaning df
    tempdf = tempdf.loc[:,~tempdf.columns.duplicated()]
    if predict:
        if len(tempdf) > 1:
            tempdf = tempdf.iloc[0:2, :]
            tempdf.drop(1,axis=0,inplace = True)
    else:
        tempdf = tempdf.iloc[1: , :]
    return tempdf

#get yfinance data
def get_hist_price(df,predict,data=None,mdata=None):
    tickerList1 = (df['symbol'].unique()).tolist()
    tickerList = (df['symbol']).tolist()
    dateList = df['date'].tolist()
    zipped = list(zip(tickerList, dateList))
    tickers = ' '.join(tickerList1)
    startyear = 2010
    startdate = (str(startyear)+'-01-01')
    outperformance = 10
    if data == None or mdata == None:
        #Send request to yfinance
        data = (pd.DataFrame(yf.download(f"{tickers}", start=f"{startdate}")))["Adj Close"]
        data.dropna(axis=0,how='all',inplace=True)
        data.to_csv("Stock Predictor 2.0/12345.csv")
        mdata = (pd.DataFrame(yf.download("SPY", start=f"{startdate}")))["Adj Close"]
        mdata.dropna(axis=0,how='all',inplace=True)
    else:
        data = pd.read_csv(f"Stock Predictor 2.0/{data}.csv",index_col="Index")
        mdata = pd.read_csv(f"Stock Predictor 2.0/{mdata}.csv",index_col="Index")

    # We will reindex to include the weekends.
    #This at the same time gives "" for tickers without value
    start_date = str(min(data.index))
    end_date = str(max(data.index))
    idx = pd.date_range(start_date, end_date)
    mdata = mdata.reindex(idx)
    data = data.reindex(idx)
    #Weekends take friday value
    mdata.ffill(inplace=True)
    data.ffill(inplace=True)
    #Store Data
    data.to_csv('Stock Predictor 2.0/Data_YF_Hist_Price.csv',index_label="Index")
    mdata.to_csv('Stock Predictor 2.0/MData_YF_Hist_Price.csv',index_label="Index")
    #Initialize dataframe
    tempdf = pd.DataFrame(columns= ['symbol','name','Buy','sector','Stock Change','Market Change','Current Stock Price','Stock Price','Past Stock Change','50-day Avg','200-day Avg',
                                    '50-day low','200-day low','50-day high','200-day high','fiftyTwoWeekHigh','fiftyTwoWeekLow',"targetLowPrice","targetMedianPrice"])
    #Obtain dates of each row
    for s,d in tqdm(zipped, desc ="Yahoo Finance API"):
        time.sleep(0.5)
        try:
            data_info = yf.Ticker(f"{s}").info
        except:
            tempdf.to_csv(f"Stock Predictor 2.0/DATA_ERROR_{s}.csv")
        date = str(d)
        future = str((dt.datetime.strptime(d, "%Y-%m-%d") + dt.timedelta(days=365)).strftime("%Y-%m-%d"))
        past = str((dt.datetime.strptime(d, "%Y-%m-%d") - dt.timedelta(days=365)).strftime("%Y-%m-%d"))
        fiftyday = str((dt.datetime.strptime(d, "%Y-%m-%d") - dt.timedelta(days=50)).strftime("%Y-%m-%d"))
        twohunday = str((dt.datetime.strptime(d, "%Y-%m-%d") - dt.timedelta(days=200)).strftime("%Y-%m-%d"))

        print(start_date)

        if len(zipped) > 1 and start_date < d:
            stockprice50d = data.loc[fiftyday:date,s].sum()/50
            stockprice200d = data.loc[twohunday:date,s].sum()/200
            print(data.loc[fiftyday:date,s])
            low50d = min(data.loc[fiftyday:date,s])
            low200d = min(data.loc[twohunday:date,s])
            high50d = max(data.loc[fiftyday:date,s])
            high200d = max(data.loc[twohunday:date,s])
            paststock = data.loc[past,s]
            currentstock = data.loc[date,s] #current in this case is the stock price at the date given not the current price
            currentmarket = mdata.loc[date] #current in this case is the S&P price at the date given not the current price
            if predict:
                furturestock = 0
                futuremarket = 0
            else:
                furturestock = data.loc[future,s]
                futuremarket = mdata.loc[future]    
        elif start_date < d:
            stockprice50d = data.loc[fiftyday:date].sum()/50
            stockprice200d = data.loc[twohunday:date].sum()/200
            low50d = min(data.loc[fiftyday:date])
            low200d = min(data.loc[twohunday:date])
            high50d = max(data.loc[fiftyday:date])
            high200d = max(data.loc[twohunday:date]) 
            paststock = data.loc[past]
            currentstock = data.loc[date]
            currentmarket = mdata.loc[date]
            if predict:
                furturestock = 0
                futuremarket = 0
            else:
                furturestock = data.loc[future]
                futuremarket = mdata.loc[future]    
        else:
            #if data out of range
            print("-- Date Out of Range--")
            print("Start Date:",start_date)
            print("Date:",d)
            print("Ticker:",s)
            low50d = 0
            low200d = 0
            high50d = 0
            high200d = 0
            stockprice50d = 0
            stockprice200d = 0
            currentstock = 1
            currentmarket = 1   
        #Calculating % price change and y value
        if predict and len(zipped) > 1:
            paststock = data.loc[past,s]
            stockchange = 1
            marketchange = 1
            buy = False
        elif predict:
            paststock = data.loc[past]
            stockchange = 1
            marketchange = 1
            buy = False
        else:
            stockchange = round(((furturestock - currentstock) / currentstock * 100), 2)
            marketchange = round(((futuremarket - currentmarket) / currentmarket * 100), 2)
            buy = outperformance +  marketchange < stockchange

        #Debug
        file = open("debug.txt","w")
        file.write(str(data_info))
        file.close()
        try:        
            row = [s,data_info['longName'],buy,data_info['sector'], stockchange, marketchange,data_info['currentPrice'],currentstock,paststock,stockprice50d,stockprice200d,low50d,low200d,high50d,high200d,data_info['fiftyTwoWeekHigh'],data_info['fiftyTwoWeekLow'],data_info["targetLowPrice"],data_info["targetMedianPrice"]]
        except:
            try:
                row = [s,data_info['longName'],buy,"", stockchange, marketchange,data_info['currentPrice'], currentstock,paststock,stockprice50d,stockprice200d,low50d,low200d,high50d,high200d,data_info['fiftyTwoWeekHigh'],data_info['fiftyTwoWeekLow'],data_info["targetLowPrice"],data_info["targetMedianPrice"]]
            except:
                row = [s,"",buy,"", stockchange, marketchange,"",currentstock,paststock,stockprice50d,stockprice200d,low50d,low200d,high50d,high200d,"","","",""]
        df_length = len(tempdf)
        tempdf.loc[df_length] = row
    return(tempdf)

#Process final dataset
def process_data(df): 
    df.sort_values(by="symbol",ignore_index=True,inplace=True)
    return df

#view dataset
def view_dataframe(savefile,df):
    if savefile:
        df = pd.read_csv(f"{savefile}.csv",index_col="Index")
    #Process info
    buy = df['Buy'].tolist()
    symbol = df['symbol'].unique()
    buypercent = sum(buy)/len(df)*100  
    percentmissing = (df.isnull().sum() * 100 / len(df)).sort_values(ascending=False)
    percentmissing = percentmissing.iloc[:11]
    #Print dataframe info
    print(df['Stock Change'].describe())
    print(f"Buy Percentage:{buypercent: .2f}%")
    print(f"Total row: {len(df)}")
    print(f"Total Unique Symbols: {len(symbol)}")
    print(f"Dataframe shape: {df.shape}")
    print(f"Top 10 Null Percentages Columns:")
    print(percentmissing)

#Run all the functions
def full_run(tickerList,savefile,predict = False,returndf = False):
    global count1
    #Test for save file
    try:
        df = pd.read_csv(f"Stock Predictor 2.0/{savefile}.csv",index_col="Index")
        tickerList = [ticker for ticker in tickerList if ticker not in df["symbol"].unique()]
        tickerList = [ticker for ticker in tickerList if ticker not in failList]
    except:
        df = pd.DataFrame()
    index = 0
    count = 0
    failed = []
    stop = False
    tempdf = pd.DataFrame()
    for ticker in tqdm(tickerList, desc ="FMP API"):
        if stop:
            break
        #Test API limit
        while True:
            row = None
            try:
                row = get_all(index,ticker,predict)
                if len(row) > 0:
                    tempdf = pd.concat([tempdf,row],ignore_index=True)
                    count += 1
                    break
                else:
                    failed.append(ticker)
                    break
            except KeyError:
                if index < len(apikeys) - 1:
                    time.sleep(3)
                    index += 1
                else:
                    print("API Keys out of requests!")
                    print(f"Count: {len(tempdf['symbol'].unique())}") #error
                    print(f"API key: {apikeys[index]}")
                    stop = True
                    break
            except:
                #If any of the API returns null
                failed.append(ticker)
                break

    #Adding buy and historic prices
    if len(tempdf) > 0:
        histpricedf = get_hist_price(tempdf,predict)
        if len(histpricedf) > 0:
            #Horizontal concat
            tempdf = pd.concat([histpricedf,tempdf], axis=1, join='inner')
            #Cleaning tempdf
            tempdf = tempdf.loc[:,~tempdf.columns.duplicated()]
        else:
            failed += tempdf['symbol'].unique().tolist()
            print(f"Nothing Added! (Failure due to YF)")
            print(f"Failed Tickers:{failed}")
            file = open('failedTickerList.txt', 'a')
            # Append failed tickers
            data = '/'.join(failed)
            if data:
                file.write('/' + data)
            file.close()
            return("Operation Failed!")
    else:
        print(f"Nothing Added!(Failure due to FMP)")
        print(f"Failed Tickers:{failed}")
        file = open('failedTickerList.txt', 'a')
        # Append failed tickers
        data = '/'.join(failed)
        if data:
            file.write('/' + data)
        file.close()
        return("Operation Failed!")
    
    #Catching historical failed symbols
    failed = failed + (tempdf[tempdf['Stock Change'].isnull()]["symbol"].unique()).tolist()
    tempdf = tempdf[tempdf['Stock Change'].notna()]
    #Final df concat
    df = pd.concat([df,tempdf],ignore_index=True)
    df = process_data(df)
    #Print status of the API
    print(f"API Key:{apikeys[index],index}")
    print(f"Total Ticker Count:{len(tempdf['symbol'].unique())}")
    print(f"Added Rows:{len(tempdf.index)}")
    print(f"Failed Tickers:{failed}")
    print(f"Final Count: {count}")
    #Saving information to files
    #Add file
    file = open("count.txt","w")
    file.write(str(count1))
    file.close()
    file = open('failedTickerList.txt', 'a')
    # Append failed tickers
    data = '/'.join(failed)
    if data:
        file.write('/' + data)
    file.close()
    if returndf:
        return(df)
    else:
        df.to_csv(f"Stock Predictor 2.0/{savefile}.csv",index_label="Index")

#Run predictions
def pred_run(tickerList,savefile = "testdata"):
    global count1
    predict = True
    #Test for save file
    tickerList = [ticker for ticker in tickerList if ticker not in failList]
    df = pd.DataFrame()
    index = 0
    count = 0
    failed = []
    stop = False
    tempdf = pd.DataFrame()
    for ticker in tqdm(tickerList):
        if stop:
            break
        #Test API limit
        while True:
            try:
                row = get_all(index,ticker,predict)
                print("row")
                print(row)
                tempdf = pd.concat([tempdf,row],ignore_index=True)
                count += 1
                break
            except KeyError:
                #Change API key
                if index < len(apikeys) - 1:
                    time.sleep(3)
                    index += 1
                else:
                    print("API Keys out of requests!")
                    print(f"Count: {count}")
                    print(f"API key: {apikeys[index]}")
                    stop = True
                    break
                pass
            except:
                #If any of the API returns null
                failed.append(ticker)
                break

    #Adding buy and historic prices
    histpricedf = get_hist_price(tempdf,predict)
    #Horizontal concat
    tempdf = pd.concat([histpricedf,tempdf], axis=1, join='inner')
    tempdf.to_csv(f"test.csv",index_label="Index")
    #Cleaning tempdf
    tempdf = tempdf.loc[:,~tempdf.columns.duplicated()]
    #Catching historical failed symbols
    failed = failed + (tempdf[tempdf['Stock Change'].isnull()]["symbol"].unique()).tolist()
    tempdf = tempdf[tempdf['Stock Change'].notna()]
    #Final df concat
    df = pd.concat([df,tempdf],ignore_index=True)
    df = process_data(df)
    #Print status of the API
    print(f"API Key:{apikeys[index],index}")
    print(f"Total Ticker Count:{len(tempdf['symbol'].unique())}")
    print(f"Added Rows:{len(tempdf.index)}")
    print(f"Failed Tickers:{failed}")
    #Saving information to files
    file = open("count.txt","w")
    file.write(str(count1))
    print(count1)
    file.close()
    df.to_csv(f"{savefile}.csv",index_label="Index")

#Change outperformance
def change_buy(outperformance,savefile):
    df = pd.read_csv(f"Stock Predictor 2.0/{savefile}.csv",index_col="Index")
    df["Buy"] = df["Stock Change"] > df["Market Change"] + outperformance
    df.to_csv(f"Stock Predictor 2.0/{savefile}.csv",index_label="Index")

#full_run(["PTR"],"test1",True)



'''
df = pd.read_csv("Stock Predictor 2.0\FMP_YF_Data_US.csv")
df.drop(columns=['symbol','name','Buy','sector','Stock Change','Market Change','Stock Price','Past Stock Change','50-day Avg','200-day Avg',
                '50-day low','200-day low','50-day high','200-day high','fiftyTwoWeekHigh','fiftyTwoWeekLow',"targetLowPrice","targetMedianPrice"])
df2 = get_hist_price(df,False,data=None,mdata=None)
tempdf = pd.concat([df2,df], axis=1, join='inner')
#Cleaning tempdf
tempdf = tempdf.loc[:,~tempdf.columns.duplicated()]
tempdf.to_csv("Stock Predictor 2.0\FMP_YF_Data_US2.csv")
'''