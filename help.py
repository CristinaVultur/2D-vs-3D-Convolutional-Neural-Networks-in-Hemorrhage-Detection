import pandas as pd
val = pd.read_csv('valid.csv')
print(val['any'].value_counts())