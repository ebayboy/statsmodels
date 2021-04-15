import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
 
#df = pd.read_csv("https://gitee.com/myles2019/dataset/raw/master/jetrail/jetrail_train.csv")
df = pd.read_csv("./jetrail_train.csv")

print("df.head():\n%s" % df.head())
print("df.tail():\n%s" % df.tail())
