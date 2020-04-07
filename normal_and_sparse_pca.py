import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
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


pca = PCA(n_components=2)
sparsepca = SparsePCA(n_components=2)

principalComponents=pca.fit(x)
principalComponents = pca.fit_transform(x)
SparseprincipalComponents = sparsepca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents, columns = ['n1', 'n2'])
SprincipalDf = pd.DataFrame(data = SparseprincipalComponents, columns = ['s1', 's2'])

principalDf.to_csv("normal.csv")
SprincipalDf.to_csv("sparse.csv")


def individual_plots():
	colors=[]
	'''
	for i in range(0,649):
		if ((i==99) or (i==143) or (i==388)):
			print(str(i)+"-->"+"n1="+str(principalComponents[:,0][i])+" "+"n2="+str(principalComponents[:,1][i]))
			colors.append("r")
		else:
			colors.append("b")	
	'''
	#plt.scatter(principalDf['n1'],principalDf['n2'],c=colors)
	plt.scatter(principalDf['n1'],principalDf['n2'])
	plt.title("Normal PCA")
	plt.savefig("NormalPCA.png")
	#plt.show()
	plt.clf()

	#plt.scatter(SprincipalDf['s1'],SprincipalDf['s2'],c=colors)
	plt.scatter(SprincipalDf['s1'],SprincipalDf['s2'])
	plt.title("Sparse PCA")
	plt.savefig("SparsePCA.png")
	#plt.show()
	plt.clf()
	
def separate_plots():
	fig, (ax1, ax2) = plt.subplots(2)
	ax1.set_title("Normal PCA")
	ax2.set_title("Sparse PCA")
	fig.suptitle('filled')
	ax1.scatter(principalDf['n1'],principalDf['n2'],c='r')
	ax2.scatter(SprincipalDf['s1'],SprincipalDf['s2'])
	plt.savefig("images/combinedb_pca.png")
	plt.show()



individual_plots()
#separate_plots()

'''
This is how you concatenate dataframes

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
print(finalDf)

'''