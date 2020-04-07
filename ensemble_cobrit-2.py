import numpy as np
import pandas as pd 
import openensembles as oe
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore')





df  = pd.read_csv("filled_encoded_data.csv")
df = df.set_index("GUID")


# Applies transformers to columns of an array or pandas DataFrame
from sklearn.compose import ColumnTransformer

# Normalize samples individually to unit norm.
# Scaling inputs to unit norms is a common operation for text classification or clustering for instance
from sklearn.preprocessing import Normalizer

# Encode categorical integer features as a one-hot numeric array.
from sklearn.preprocessing import OneHotEncoder

#If you want to add plots to your Jupyter notebook, then %matplotlib inline is a standard solution.
#%matplotlib inline


#np.random.seed(a_fixed_number) every time you call the numpy's other random function, the result will be the same
#However, if you just call it once and use various random functions, the results will still be different:
np.random.seed(0) #this helps to establish the same dataset and functionality, but is not required


# Open a csv file and convert to dataframe object


#df = pd.read_csv('Data/DataGranulatedGCSNoPT.csv') -----


# convert dataframe to oe dataobject
# second argument is the  number of columns


d = oe.data(df, [i for i in range(1, len(df.columns)+1)])




'''
WHAT NEEDS TO BE ADDED FOR FUTURE USAGE FROM OTHERS:
    a) After loading data
        Ensure everything is either normalized or handle in this cell before creating the ensemble
        
    b) Ensure categorical features are encoded
    
    Code cell accidently deleted for encoding but pandas has useful tool for easy encoding
    https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html

'''


# pass oe dataobj to cluster class
c  = oe.cluster(d) #instantiate an object so we can get all available algorithms


# Call this to list all algorithms currently available in algorithms.py
# These are the algorithms available
# [kmeans,spectral,agglomerative, DBSCAN,HDBSCAN,AffinityPropagation,Birch,MeanShift,GaussianMixture,] 
a = c.algorithms_available()


# returns Keys equal to parameters {K, linkages, distances} and values as lists of algorithms that use that key as a variable
# Example : 
paramsC = c.clustering_algorithm_parameters() #here we will rely on walking through 
    
# remove DBSCAN -- this does very well on unstructured data, we want to ask if we can use poorly performing algorithms 
# to identify if there isn't structure.
# 'DBSCAN', 'Birch', 'GaussianMixture', 'HDBSCAN', 'MeanShift', 'AffinityPropagation','kmeans', 'agglomerative', 'spectral'
algorithmsToRemove = ['DBSCAN', 'Birch','HDBSCAN','AffinityPropagation','MeanShift']

for algToRemove in algorithmsToRemove:
    del a[algToRemove]


# Get all the algorithms that take linkage

takesLinkages = paramsC['linkage']


# Get the algorithms that takes distance


takesDistances = paramsC['distance']


# Take all the algorithms that take K
takesK = paramsC['K']



#setup the parameters of clustering here, algorithms are set by algorithms_available
# K is [2,3,4,5,6,7,8,9,10]
#K = range(2, 11, 1)
K = list(range(2,11,1))
linkages = ['ward'] 

# These are the types of distances
distances = ['euclidean','manhattan']


# Create an ensemble: sweep K, distance metrics
c = oe.cluster(d)


for data_source in d.D.keys(): #if there were transformations in d.D
    #print(data_source)
    # This will be the list of algorithms available in the list
    for algorithm in list(a.keys()): #linkage is only for agglomerative, which also accepts K and distances, so handle that here
    # check if the algorithm takes K
        if algorithm in takesK:
            # Loop through all the K's we want.That is 2,3,4,5,6,7,8,9,10
            for k in K:
                # Check if the algorithms takes distance
                if algorithm in takesDistances:
                    # Check if the algorithm takes linkages
                    if algorithm in takesLinkages:
                        # Go through all the linkages
                        for linkage in linkages:
                            if linkage == 'ward':

                                #out_name = '_'.join([data_source, algorithm, linkage, str(k)]) 
                                out_name = '_'.join([algorithm,linkage, str(k)])
                                c.cluster(data_source, algorithm, out_name, K=k, Require_Unique= True, linkage=linkage)
                                
                                
                            # check if linkage is not ward
                            else:
                                # go through the distances [euclidean,L1,l2]
                                for dist in distances:
                                    out_name = '_'.join([algorithm,dist, linkage, str(k)])

                                    # Create the cluster with the data source,algorithm,output name,number of K's 
                                    c.cluster(data_source, algorithm, out_name, K=k, Require_Unique= True, linkage=linkage, distance=dist)


                    # Algorithm does not take linkages                
                    else:
                        # Go through all the distances
                        for dist in distances:
                            out_name = '_'.join([algorithm, dist, str(k)])
                            c.cluster(data_source, algorithm, out_name, K=k, Require_Unique= True, distance=dist)
                # Algorithm does not take distance
                else:
                    out_name = '_'.join([algorithm, str(k)])
                    c.cluster(data_source, algorithm, out_name, K=k, Require_Unique= True)


        # Algorithm does not take K            
        else: # does not take K

            # Check if algorithm  takes distance
            if algorithm in takesDistances:
                    for dist in distances:
                        out_name = '_'.join([algorithm, dist])
                        c.cluster(data_source, algorithm, out_name, Require_Unique= True, distance=dist)
            # Algorithm that does not take distance
            else:
                out_name = '_'.join([algorithm])
                c.cluster(data_source, algorithm, out_name, Require_Unique= True)





# Apply PCA on dataset by 2D
#d.transform("parent","PCA","pca",n_components=2)
#d.plot_data("pca",title="original PCA data",class_labels =iris.target)
#plt.savefig("Original PCA Dataset 2D")
#plt.show()
#print(iris.target)

result_df = pd.DataFrame()


# Plot and save each of the clustering solutions (Each clustering solution brings us labels) using the original dataset
cluster_solution_names = c.labels.keys()
for name in cluster_solution_names:
    print(name)
    # For actual solution
    mini_df = pd.DataFrame()
    mini_df[""] = c.labels[name]
    mini_df[""] = mini_df[""]+1
    mini_df.to_csv("All Results\\Folder 2\\Clustering Results\\"+name+".csv",index=False,header=False)

    result_df[name] = c.labels[name]
    result_df[name] = result_df[name]+1
    
   
    
    # For iris with same name
    #mini_df = pd.DataFrame() 
    #mini_df[""] = iris.target
    #mini_df[""] = mini_df[""]+1
    #mini_df.to_csv("Labels Only\\"+name+".csv",index=False,header=False)
    




    #d.plot_data("pca",fig_num=1,class_labels=c.labels[name],title=name)
    #plt.savefig("images\\"+name+".jpg")
    #plt.show()
    

# Mixture Model 
for i in range(2,11):
    print("Mixture Model "+str(i))
    mixture_model = c.mixture_model(i)
    mixture_model_labels = mixture_model.labels["mixture_model"]



    # For mixture model solution (Mixture model already clusters starting from 1.So we dont need to add)
    mini_df = pd.DataFrame()
    mini_df["mixture_model_"+str(i)] = mixture_model_labels
    #mini_df[""] = mini_df[""]
    mini_df.to_csv("All Results\\Folder 2\\Clustering Results\\mixture_model_{}.csv".format(str(i)),index=False,header=False)

    result_df["mixture_model_"+str(i)] = mixture_model_labels




    # For iris with same name
    #mini_df = pd.DataFrame() 
    #mini_df[""] = iris.target
    #mini_df[""] = mini_df[""]+1
    #mini_df.to_csv("Labels Only\\mixture_model.csv",index=False,header=False)

    #d.plot_data("pca",fig_num=1,class_labels=mixture_model_labels,title="mixture_model_{}".format(str(i)))
    #plt.savefig("images\\mixture_model_{}.jpg".format(str(i)))


# Graph Closure
# Indicate the thresholds we want
#closure_thresholds = [0.5,0.6,0.7,0.8]
closure_thresholds = np.arange(0.5,0.8,0.02)
for threshold in closure_thresholds:
    print("Graph Closure threshold "+str(threshold))
    # apply current threshold on closure and save the graph
    closure_model = c.finish_graph_closure(threshold=threshold)
    
    # Get the array of labels decided by "graph closure" and "threshold"
    closure_labels = closure_model.labels["graph_closure"]
       
    # For actual solution
    mini_df = pd.DataFrame()
    mini_df[""] = closure_labels
    mini_df[""] = mini_df[""] + 1
    mini_df.to_csv("All Results\\Folder 2\\Clustering Results\\Graph_closure threshold_{}.csv".format(str(threshold)),index=False,header=False)

    result_df["Graph_closure threshold_{}.csv".format(str(threshold))] = closure_labels
    result_df["Graph_closure threshold_{}.csv".format(str(threshold))] = result_df["Graph_closure threshold_{}.csv".format(str(threshold))]+1


result_df.to_csv("All Results\\Folder 2\\Total Results\\total_results.csv")




