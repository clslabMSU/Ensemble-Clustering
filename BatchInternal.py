 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:03:58 2017

@author: thy1995

MODIFED PATHS FOR DEBUGGING ON 8/29/2019 BY ZJR
"""

from os.path import isfile, join
from os import listdir
import numpy as np
import csv

from fileOP import writeRows
import internal_validation
from sklearn.metrics import silhouette_score
from resultOP import table_result

#dataFolder = "\\\\EGR-1L11QD2\\CLS_lab\\DEBUG_CV_ERROR\\Data Comparisons\\All Datasets\\"
dataFolder = "dataset\\"             # (pima.csv)            (filled_encoded) (rename to data.csv)
labelFolder = "clustering result\\"    # (pimaSPectral.csv)	 (mixture_model)   (add data to the name) 
internalFolder = "internal result\\"			  # (pimaSpectral-internal.csv)	

import os.path

dataFiles = [(dataFolder + f) for f in listdir(dataFolder) if isfile(join(dataFolder, f))]
labelFiles = [(labelFolder + f) for f in listdir(labelFolder) if isfile(join(labelFolder, f))]

counter = 0
for data_file_name in dataFiles:

    data_peak = np.recfromcsv(data_file_name, delimiter = ',') # peak through data to see number of rows and cols

    num_cols = len(data_peak[0])
    num_rows = len(data_peak)
    data  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col
    
    
    with open(data_file_name) as csvfile:
        row_index = 0
        reader= csv.reader(csvfile)
        for row in reader:
            for cols_index in range(num_cols):
                data[row_index][cols_index]= row[cols_index]
            row_index+=1
    
    target= data_file_name.split("\\")[-1].split(".csv")[0]
    targets = [i for i in labelFiles if i.find(target) != -1]
    #print(targets)
    
    scatL = []
    distL = []
    comL = []
    sepL = []
    for label_file_name in targets:
        print("current label", label_file_name)        
        name = label_file_name.split(".")[0].split("\\")[-1]
        exist = os.path.exists(internalFolder + name + "_internal.csv")
        if exist:
            continue
        
        data_peak = np.recfromcsv(label_file_name, delimiter = ',') # peak through data to see number of rows and cols

        num_cols = len(data_peak[0])
        num_rows = len(data_peak)
        label  = np.zeros([num_rows+1, num_cols]) # num_cols - 1 means skip label col

        with open(label_file_name) as csvfile:
            row_index = 0
            reader= csv.reader(csvfile)
            for row in reader:
                for cols_index in range(num_cols):
                    label[row_index][cols_index]= row[cols_index]
                row_index+=1
        
        label = label.T
        
        unique_list=[]
        for d_column in label:
            num_k = np.unique(d_column)
            unique_list.append("k"+str(int(max(num_k))))
            inter_index = internal_validation.internalIndex(len(num_k))
            

            print(data)
            print(d_column)
            
            scat , dis = inter_index.SD_valid(data, d_column)
            com , sep = inter_index.CVNN(data, d_column)
            scatL.append(scat)
            distL.append(dis)
            comL.append(com)
            sepL.append(sep)
        result_over_k = []
        for i in range(len(label)):
            d_column = label[i]
            num_k = np.unique(d_column)
            result = []
            inter_index = internal_validation.internalIndex(len(num_k))
            result.append(silhouette_score(data, d_column, metric = 'euclidean'))
            result.append(inter_index.dbi(data, d_column))
            result.append(inter_index.xie_benie(data, d_column))
            result.append(inter_index.dunn(data, d_column))
            #result.append(inter_index.CH(data, d_column))
            #result.append(inter_index.I(data, d_column))
            result.append(inter_index.SD_valid_n(scatL, distL, i))
            #result.append(inter_index.SDbw(data, d_column))
            result.append(inter_index.CVNN_n(comL, sepL, i))
            
            result_over_k.append(result)
        to_export = np.array(result_over_k).T
        print(unique_list)
        print(['k' + str(i) for i in range(2, len(to_export[0]) + 2 )])        
        #a = ['k' + str(i) for i in range(2, len(to_export[0]) + 2 )]
        a=unique_list
        to_export = table_result(to_export,[a] ,[['','Sil', 'Db', 'Xb', 'Dunn',"SD","CVNN"]])                       
        name = label_file_name.split(".")[0].split("\\")[-1]
        writeRows(internalFolder + name + "_internal.csv" , to_export)
    counter = counter + 1