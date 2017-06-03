import pandas as pd 
import quandl
import math
df=quandl.get('WIKI/GOOGL')
#df = df.ix[:,0:4]
#weather={'day':['1','4','7','8'],'temp':[22,44,5,1],'wind':[4,6,3,2]}
#weather['work']=[4,1,2,3]
#df=pd.DataFrame(weather)
#df['gfg']=df['work']-df['temp']
df['HL']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100
df['PCT']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100
dfs=df[['Adj. High','Adj. Low','Adj. Close','Adj. Open','HL','PCT']]
gg= 'Adj. Close' 
#dfs.fillna(-999999,inplace=True)
fore=int(math.ceil(0.1*len(dfs)))
dfs['label']=dfs[gg].shift(-1)
print(dfs.head())