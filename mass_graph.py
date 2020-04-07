import pandas as pd 
import matplotlib.pyplot as plt 
import time
# get pca normal

def isomap():
	dfn = pd.read_csv("isomap.csv")  # takes (normal.csv) or (sparse.csv) or (isomap_components.csv) or (tsne.csv)
	x = dfn["c1"].tolist()			 # takes (n1) or (s1)  or (c1)   or (t1)
	y = dfn["c2"].tolist()			 # takes (n2) or (s2)  or (c2)	 or (t2)
	dft = pd.read_csv("target.csv")	 # This is the total_consensus.csv (Contains Mixture Model and Graph Closure Results)
	columns = dft.columns
	for i in columns:
		plt.scatter(x,y,c=dft[i])
		plt.savefig("images\\isomap\\"+i+"_isomap.PNG")  # (_normalPCA.PNG) or (_sparsePCA.PNG)
		plt.clf()
		

def tsne():
	dfn = pd.read_csv("tsne.csv")  # takes (normal.csv) or (sparse.csv) or (isomap_components.csv) or (tsne.csv)
	x = dfn["t1"].tolist()			 # takes (n1) or (s1)  or (c1)   or (t1)
	y = dfn["t2"].tolist()			 # takes (n2) or (s2)  or (c2)	 or (t2)
	dft = pd.read_csv("target.csv")	 # This is the total_consensus.csv (Contains Mixture Model and Graph Closure Results)
	columns = dft.columns
	for i in columns:
		plt.scatter(x,y,c=dft[i])
		plt.savefig("images\\tsne\\"+i+"_tsne.PNG")  # (_normalPCA.PNG) or (_sparsePCA.PNG)	
		plt.clf()

def normal():
	dfn = pd.read_csv("normal.csv")  # takes (normal.csv) or (sparse.csv) or (isomap_components.csv) or (tsne.csv)
	x = dfn["n1"].tolist()			 # takes (n1) or (s1)  or (c1)   or (t1)
	y = dfn["n2"].tolist()			 # takes (n2) or (s2)  or (c2)	 or (t2)
	dft = pd.read_csv("target.csv")	 # This is the total_consensus.csv (Contains Mixture Model and Graph Closure Results)
	columns = dft.columns
	for i in columns:
		plt.scatter(x,y,c=dft[i])
		plt.savefig("images\\normal\\"+i+"_normalPCA.PNG")  # (_normalPCA.PNG) or (_sparsePCA.PNG)
		plt.clf()

def sparse():
	dfn = pd.read_csv("sparse.csv")  # takes (normal.csv) or (sparse.csv) or (isomap_components.csv) or (tsne.csv)
	x = dfn["s1"].tolist()			 # takes (n1) or (s1)  or (c1)   or (t1)
	y = dfn["s2"].tolist()			 # takes (n2) or (s2)  or (c2)	 or (t2)
	dft = pd.read_csv("target.csv")	 # This is the total_consensus.csv (Contains Mixture Model and Graph Closure Results)
	columns = dft.columns
	for i in columns:
		plt.scatter(x,y,c=dft[i])
		plt.savefig("images\\sparse\\"+i+"_sparsePCA.PNG")  # (_normalPCA.PNG) or (_sparsePCA.PNG)
		plt.clf()

isomap()
tsne()
normal()
sparse() 
