"""
author:@dacosta
date:31-10-2019
"""


from sklearn.manifold import Isomap
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "processed_encoded.csv"
# load dataset into Pandas DataFrame


#df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
df = pd.read_csv(url)

# Separating out the features
#x = df.loc[:, features].values
df = df.drop(["GUID"],axis=1)

x = df.values
# Separating out the target
#y = df.loc[:,['target']].values


# Standardizing the features
x = StandardScaler().fit_transform(x)
# scale here 

# initialize isomap object 
embedding = Isomap(n_components = 2)
components = embedding.fit_transform(x)

isomapdf = pd.DataFrame()

isomapdf["c1"] = components[:,0]
isomapdf["c2"] = components[:,1]

colors = []
'''
for i in range(0,649):
	if ((i==99) or (i==143) or (i==388)):
		print(str(i)+"-->"+"c1="+str(components[:,0][i])+" "+"c2="+str(components[:,1][i]))
		colors.append("r")
	else:
		colors.append("b")

'''
isomapdf.to_csv("isomap.csv",index=False)


#plt.scatter(components[:,0],components[:,1],marker="*")
#plt.show()
# print(components)


plt.scatter(components[:,0],components[:,1])
plt.title("isomap")
plt.savefig("isomap.jpg")
