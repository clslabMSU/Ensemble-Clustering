# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:19:22 2019

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:36:31 2017

@author: thy1995
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 14:19:36 2017

@author: thy1995
"""

from scipy.stats import weightedtau, pearsonr, spearmanr
import numpy as np
import csv
from os.path import isfile, join
from os import listdir
from fileOP import writeRows, new_name
from resultOP import table_result
import os
from itertools import combinations
import re

def makeFolder(addr):
    if not os.path.exists(addr):
        os.makedirs(addr)

# List of headers of a CSV file
header = [["Sil", "Db", "Xb", "Dunn", "CH", "I", "SD", "SDb_w", "CVNN"]]

# List of indexes of a csv file
att = [['','NMI', "Adjusted Rand", "Accuracy", "Jaccard"]]


#  
signature = 'real'

truthDict = {
        "avila": ["D-26", 12, 10, 20867],
        "breast " : ["D-17", 6, 9, 106],
        "breast_w" : ["D-06",2, 30, 569],
        "ecoli" : ["D-23",8,7, 336],
        "glass" : ["D-20",7, 9, 214],
        "haberman" : ["D-01",2, 3, 306],
        "heart" : ["D-08",2 ,44, 267],
        "Liver" : ["D-03",2, 10, 597],
        "ionosphere" : ["D-07",2, 34, 351],
        "iris" : ["D-10",3,4,150],
        "satellite" : ["D-04",2,12,310],
        "Musk1" : ["D-09",166, 476],
        "page blocks" : ["D-16",5, 10, 5473],
        "parkinsons data" : ["D-05", 2, 22, 195],
        "pima" : ["D-02", 8, 768],
        "red wine" : ["D-18",6 ,11, 1599],
        "seeds" : ["D-12",3, 7, 210],
        "segmentation" : ["D-22",7, 19, 2310],
        "spine" : ["D-04", 2, 12, 310],
        "user knowledge" : ["D-14",4 ,5, 403],
        "vehicle" : ["D-15",4, 18, 846],
        "vertebral" : ["D-11",3,6,310],
        "vowel context" : ["D-25",11, 10, 990],
        "white wine" : ["D-21",7,11,4898],
        "WINE data" : ["D-13",3, 13, 178],
        "yeast data" : ["D-24", 10, 8, 1484]
        }


# path we want to save our result
savefolder =  "D:\\CLS_lab\\UCI Machine Learning Repo Datasets12\\synthetic\\final\\" 

# path of internal files
internal_folder = "D:\\CLS_lab\\UCI Machine Learning Repo Datasets12\\synthetic\\final\\internal\\"

# path of external files
external_folder = "D:\\CLS_lab\\UCI Machine Learning Repo Datasets12\\synthetic\\final\\external\\"       


internal_file = [(internal_folder + f) for f in listdir(internal_folder) if isfile(join(internal_folder, f))]
realname = [t.split("\\")[-1].replace("internal.csv","") for t in internal_file]
external_file = [external_folder + t + "external.csv" for t in realname ]


if len(internal_file) != len(external_file) or len(internal_file) == 0:
    print("Not Equal")


# This returns ["Sil", "Db", "Xb", "Dunn", "CH", "I", "SD", "SDb_w", "CVNN"]    
header_1d = header[0]
att_1d = ['NMI', "Adjusted Rand", "Accuracy", "Jaccard"]


#data_peak = np.recfromcsv(internal_file, delimiter = ',') # peak through data to see number of rows and cols

# This will contain the boxes of internal runs
data_internal_list  = [] # num_cols - 1 means skip label col

# This will contain all boxes of internal runs (such as irisSpectralinternal)
data_external_list = []
internal_row_c = 0
external_row_c = 0
total_tau = []
total_pear = []
total_spear = []

corr_return_tau = []
corr_return_pear = []
corr_return_spear = []

valueL = []         # This will contain all spearmans values.Note that this value depends on pearson's conditions
dataL = []
algorithmL = []
interL = []
exterL = []
centroidL= []
featureL = []
removeKeywords = ['Kmeans', 'Spectral', 'Complete', 'Average', 'Ward']

for internal_file_name in internal_file:
    data_internal = np.loadtxt(internal_file_name, delimiter = ",", dtype = str)

    # Read a csv file and return everything starting from position 1
    data_internal = data_internal[1:]        

    # This returns the box(internal validation) without the (K2,K3,K4) and (Sil,SD) boxes
    data_internal = (data_internal.T[:-1]).T
    data_internal_list.append(data_internal.astype(float))

for external_file_name in external_file:
    data_external = np.loadtxt(external_file_name, delimiter = ",", dtype = str)
    # I think this should be modified because our data is structured differently (test flipping methods)
    data_external = np.array(data_external[1:])
    data_external = (data_external.T[:-1]).T
    data_external_list.append(data_external.astype(float))


# Go through the list which holds the data_external_list
for dataset_index, intern in enumerate(data_internal_list):

    # get the file of the current box (internal file) and find its corresponding (external file) == Return the position of the external file
    extern = [i for i in range(len(external_file)) if external_file[i].split("\\")[-1].replace("external", "internal") == internal_file[dataset_index].split("\\")[-1]][0]
    
    # Since we got the position of the external file,we get the position of the current box(external)
    extern = data_external_list[extern] # This will be one item

    correlation_return_tau = []
    correlation_return_pearson = []
    correlation_return_spearmann = []


    for extern_index_i in range(len(extern)): #
        temp_tau = []        # this contains values based on computation of two K2 columns (intern and external validation)
        temp_pearson = []    # this contains values.But it checks for some conditions before that value is added
        temp_spearmann = []  # this contains values.But the value is based on what pearson says   
        for intern_index_i in range(len(intern)):
            # pass in k2 (Box)(internal) and K2 (Box)(external)
            temp_tau.append(weightedtau(extern[extern_index_i], intern[intern_index_i]).correlation)

            # if K2 Box(external) contains only 1 unique number (like 0) (all scores were the same)
            if len(np.unique(extern[extern_index_i])) == 1:
                temp_pearson.append(0)
                value = 0
            else:    
                temp_pearson.append(pearsonr(extern[extern_index_i], intern[intern_index_i])[0]) 
                value  = spearmanr(extern[extern_index_i], intern[intern_index_i]).correlation
            temp_spearmann.append(value)
            
            valueL.append(value)
            interL.append(header_1d[intern_index_i])  # eg. Sil -> go to CH on next iteration
            exterL.append(att_1d[extern_index_i])     # eg. Jaccard  -> go to NMI on next iteration
            dataname = internal_file[dataset_index]
            for r in removeKeywords:
                if dataname.find(r) != -1:
                    datal = r
                    dataname = dataname.replace(r ,'')
                    dataname = dataname.replace('internal' ,'')
            dataname = dataname.split('.')[0].split('\\')[-1]
#            cen = ''
#            for c in centroid:
#                if dataname.find(c)!= -1:
#                    cen = c
            attributes = re.findall(r'\d+', dataname)
            _, n_feature, n_cluster = attributes
            dataL.append(dataname.replace("data", ""))
            algorithmL.append(datal)
            featureL.append(n_feature)
            centroidL.append(n_cluster)
   
ALL = []

ALL.append(valueL)
ALL.append(interL)
ALL.append(exterL)
ALL.append(dataL)
ALL.append(centroidL)
ALL.append(featureL)
ALL.append(algorithmL)

ALL = np.array(ALL).T.tolist()
ALL.insert(0, ["Spearman", "IVM", "EVM", "Dataset","Cluster", "Feature", "Algorithm"])
writeRows(savefolder +  'Synthetic.csv', ALL)