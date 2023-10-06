import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("traf.csv")
df=df.drop('date',axis=1)
df = pd.concat([df.drop('city',axis=1),pd.get_dummies(df.city)],axis=1)
df = pd.concat([df.drop('shop',axis=1),pd.get_dummies(df.shop)],axis=1)
df = pd.concat([df.drop('brand',axis=1),pd.get_dummies(df.brand)],axis=1)
df = pd.concat([df.drop('container',axis=1),pd.get_dummies(df.container)],axis=1)
correlations = df.corr()['quantity'].abs()
sorted_correlations = correlations.sort_values()
num_cols_to_drop = int(0.8*len(df.columns))
cols_to_drop = sorted_correlations.iloc[:num_cols_to_drop].index
df_dropped = df.drop(cols_to_drop,axis=1)
train_df,test_df = train_test_split(df_dropped,test_size=0.2)
train_x = train_df.drop('quantity',axis=1)
train_y = train_df['quantity']
test_x = test_df.drop('quantity',axis=1)
test_y = test_df['quantity']
forest = RandomForestClassifier()
forest.fit(train_x,train_y)
print(forest.score(test_x,test_y))