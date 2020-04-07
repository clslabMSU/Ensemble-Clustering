import pandas as pd 
import operator
import copy


name1 = "Folder 2"
name2 = "Folder 3"
name3 = "merged_mixture_model"
name4 = "merged"
name5 = "mixture_model"

name = name4



# Please check the validation indeices results and enter the column number of the indices list found at the end
df = pd.read_csv(name+".csv",index_col=38)

weights = {
	"Sil":1,
	"Db":-1,
	"Xb":-1,
	"Dunn":1,
	"CH":1,
	"I":1,
	"SD":-1,
	"SDb_w":-1,
	"CVNN":-1
}

r = 5;

# names of all columns
columns = df.columns.tolist()


# name of all rows
rows = df.index.tolist()


CVM_RANKS = { }
COLUMN_RANKS = { }
CVM_WEIGHT_RANKS = { } # HOLDS 5,4,3,2,1 and 0 for the other K


# Rank each CVM according to the K2,K3 etc
for cvm in rows:
	CVM_RANKS[cvm] = {}
	for column in columns:
		# get the score of the k multiplied by the weight
		# get the score assigned to K2 by SD 
		score = df.ix[cvm][column] * weights[cvm]
		CVM_RANKS[cvm][column] = score 


# sort each index according to the values from biggest to lowest
for cvm in rows:
	CVM_RANKS[cvm] = sorted(CVM_RANKS[cvm].items(),key=operator.itemgetter(1),reverse=True)


# for each cvm Dunss,SD..get the weighted rank for all k2,k3...
for cvm in rows:
	CVM_WEIGHT_RANKS[cvm] = { }
	score_counter = r 
	for member in list(dict(CVM_RANKS[cvm]).keys()):
		# member = k2 or k3 or etc
		CVM_WEIGHT_RANKS[cvm][member] = max(score_counter,0)
		score_counter = score_counter - 1 

#print(CVM_RANKS["Sil"])
#print(CVM_WEIGHT_RANKS["Sil"])


# for each k2,k3...get the weighted rank for all cvms
# initialize all column ranks to 0

for column in columns:
	COLUMN_RANKS[column] = 0


# for k2, get all scores from SD,XB etc and sum together.This becomes the score for k2
for column in columns:
	for cvm in CVM_WEIGHT_RANKS:
		COLUMN_RANKS[column]+=CVM_WEIGHT_RANKS[cvm][column] 		

print(COLUMN_RANKS)

# get the rth best columns 
#BEST_COLUMNS = sorted(COLUMN_RANKS.items(),key=operator.itemgetter(1),reverse=True)[0:r]

# get all the columns because all are best.We can take out what we dont need
BEST_COLUMNS = sorted(COLUMN_RANKS.items(),key=operator.itemgetter(1),reverse=True)[0:]
print(BEST_COLUMNS)



list_of_k = list(dict(BEST_COLUMNS).keys())
print(list_of_k)

# generate report

report_columns = copy.copy(rows)
report_columns.insert(0, "Weighted Score")
report = pd.DataFrame(columns=report_columns,index = list_of_k)




for k in list_of_k:
	for cvm in rows:
		report.at[k,"Weighted Score"] = COLUMN_RANKS[k]

		report.at[k,cvm] = dict(CVM_WEIGHT_RANKS[cvm])[k] 

report.index.name = "Clustering Output"

report.to_csv(name+" results.csv")


'''
print(df["k2"])
print(df.ix["Sil"]["k2"])
print(df.columns[0])
print(df.index[0])
# '''

