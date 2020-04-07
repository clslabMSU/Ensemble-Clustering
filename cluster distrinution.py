import pandas as pd 

names = ["graph_closure","mixture_model"]
for name in names:
	df = pd.read_csv(name+".csv")
	with pd.ExcelWriter(name+"_distribution.xlsx",engine="xlsxwriter") as writer :
		counter=0
		for column in df:
			dictionary = dict(df[column].value_counts())
			df_temp = pd.Series(data=dictionary)
			#df_temp.to_csv(str(columncsv),header = str(column))
			
			
			df_temp.index.name = "Labels for "+column
			df_temp=df_temp.reset_index(name="Label count")
			print(df_temp)
			#exit()
			df_temp.to_excel(writer,sheet_name=str(name),startrow=counter,startcol=0,index=True)

			if (sorted(dict(df[column].value_counts()).values())[0]<=5):
				df=df.drop([column],axis=1) 

			if (len(dictionary)==1):
				df = df.drop([column],axis=1)


			counter = counter + len(dictionary) +5

		df.to_csv(name+"_updated.csv",index=False)






