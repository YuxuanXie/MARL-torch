import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("vf.csv")

data_trian = pd.DataFrame()
data_test = pd.DataFrame()

for _, group in data.groupby("horizon"):
  train, test = train_test_split(group, test_size=0.3)
  data_trian = pd.concat([data_trian, train], ignore_index=True)
  data_test = pd.concat([data_test, test],ignore_index=True)

# print(data_trian.to_string())
# print(data_test.to_string())

data_trian.to_csv("trian.csv", index=False)
data_test.to_csv("test.csv", index=False)
