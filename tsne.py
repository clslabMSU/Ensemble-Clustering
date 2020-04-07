"""
author:@dacosta
date:31-10-2019
"""


from sklearn.manifold import TSNE
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
#x = StandardScaler().fit_transform(x)		# No need to standardize
# scale here 

# initialize isomap object 
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
components = tsne.fit_transform(x) 



tsnedf = pd.DataFrame()

tsnedf["t1"] = components[:,0]
tsnedf["t2"] = components[:,1]

colors = []

'''
for i in range(0,649):
	if ((i==99) or (i==143) or (i==388)):
		print(str(i)+"-->"+"c1="+str(components[:,0][i])+" "+"c2="+str(components[:,1][i]))
		colors.append("r")
	else:
		colors.append("b")
'''
tsnedf.to_csv("tsne.csv",index=False)


#plt.scatter(components[:,0],components[:,1],marker="*")
# plt.show()
# print(components)


plt.scatter(components[:,0],components[:,1])
plt.title("tsne")
plt.savefig("tsne.jpg")