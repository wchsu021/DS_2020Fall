import pandas as pd
import numpy as np

def read_egonet_file(filename):
	
	friend_list = []
	mutual_friend_list = dict()
	for line in open(filename):
		
		e1, es = line.split(':')
		es = es.split()
		friend_list.append(e1)
		
		m_friend = []
		for e in es:
			if e == e1: continue
			m_friend.append(e)
		
		mutual_friend_list[e1] = m_friend.copy()
	
	return friend_list, mutual_friend_list

def read_circle_file(filename):

	circle_list = dict()
	for line in open(filename):
		
		e1, es = line.split(':')
		es = es.split()
		e1 = e1[6:]
		
		circle = []
		for e in es:
			circle.append(e)

		circle_list[e1] = circle.copy()

	return circle_list

def read_feature_list():

	filename = 'featureList.txt'
	feature_list = []
	for line in open(filename):

		feature = line[:-1]
		feature_list.append(feature)

	return feature_list

def read_person_feature(person_list, feature_list):

	filename = 'features.txt'
	person_feature_list = []
	for line in open(filename):
		person_info = line.split()
		if person_info[0] in person_list:
			# print(person_info[0])
			person_info_list = [None] * 57
			for info in person_info[1:]:
				temp = info.split(';')
				value = temp[-1]
				key = ';'.join(temp[:-1])
				# print(key, value)
				f_idx = feature_list.index(key)
				if person_info_list[f_idx] != None:
					# print("Repeat!", key)
					person_info_list[f_idx] = person_info_list[f_idx] + ',' +value
				else:
					person_info_list[f_idx] = value
			person_feature_list.append(person_info_list)
	person_feature_df = pd.DataFrame(person_feature_list, columns = feature_list)
	# return person_feature_list
	person_feature_df.set_index('id', inplace=True)
	return person_feature_df

def feature_count(df):

	feature_counter = []
	cols = list(df)
	
	for col in cols:
		
		feature_dict = dict()
		
		for data in df[col]:
			
			if data != None:
				if ',' in data:
					data_list = data.split(',')
				else:
					data_list = [data]
				
				for d in data_list:
					if d in feature_dict.keys():
						feature_dict[d] += 1
					else:
						feature_dict[d] = 1
						
		feature_counter.append(feature_dict)


	return feature_counter

def circle_label(df, circle_list):

	print(df.shape)
	circle = []
	for index, row in df.iterrows():
		
		circle_num = search_circle(index, circle_list)
		if len(circle_num) == 0:
			circle.append(None)
		else:
			circle.append(','.join(circle_num))

	df['circle'] = circle
	return df

def search_circle(pid, circle_list):

	circle_num = []
	circle_key_list = circle_list.keys()
	for ck in circle_key_list:
		if pid in circle_list[ck]:
			circle_num.append(ck)
	return circle_num

def gen_adjacency_weight(friend_list, feature_list, feature_weight, person_feature_df):

	l = len(friend_list)
	adjacency_weight = np.zeros((l,l), dtype = np.float) 
	for i in range(l):
		for j in range(l):
			adjacency_vector = []
			for f in feature_list:
				
				fi = person_feature_df.at[friend_list[i],f]
				fj = person_feature_df.at[friend_list[j],f]
				
				if i != j:
					if fi == None or fj == None:
						adjacency_vector.append(0)
					else:
						fi = fi.split(',')
						fj = fj.split(',')
						
						check = False
						for value in fi:
							if value in fj:
								check = True
								break 
						if check == True:
							adjacency_vector.append(1)
						else:
							adjacency_vector.append(0)
				else:
					adjacency_vector.append(0)
			# print(adjacency_vector, len(adjacency_vector))
			weight = np.inner(feature_weight, adjacency_vector)/len(feature_weight)
			# print("weight: ", weight)
			adjacency_weight[i][j] = weight
			pass
	return adjacency_weight
	pass

def gen_adjacency_weight_v2(friend_list, feature_list, feature_weight, person_feature_df):

	l = len(friend_list)
	adjacency_weight = np.zeros((l,l), dtype = np.float) 
	for i in range(l):
		for j in range(l):
			adjacency_vector = []
			for f in feature_list:
				
				fi = person_feature_df.at[friend_list[i],f]
				fj = person_feature_df.at[friend_list[j],f]
				
				if i != j:
					if fi == None or fj == None:
						adjacency_vector.append(np.nan)
					else:
						fi = fi.split(',')
						fj = fj.split(',')
						
						check = False
						for value in fi:
							if value in fj:
								check = True
								break 
						if check == True:
							adjacency_vector.append(1)
						else:
							adjacency_vector.append(0)
				else:
					adjacency_vector.append(0)
			# print(adjacency_vector, len(adjacency_vector))
			# print(feature_weight*adjacency_vector)
			weight = np.nanmean(feature_weight*adjacency_vector)
			# print("weight: ", weight)
			adjacency_weight[i][j] = weight
			pass
	return adjacency_weight
	pass

if __name__ == '__main__':
	
	target = 239
	ego_file = 'egonets/'+str(target)+'.egonet'
	circle_file = 'Training/'+str(target)+'.circles'
	
	## friend_list
	friend_list, mutual_friend_list = read_egonet_file(ego_file)
	print("friend_list", len(friend_list))
	print(friend_list)

	## feature list
	feature_list = read_feature_list()

	## circle
	circle_list = read_circle_file(circle_file)
	print("="*40)
	print("circle")
	for key, value in circle_list.items():
		print(key, value)
		pass

	## feature dataFrame
	print("="*40)
	print("Feature of Each person")
	person_feature_df = read_person_feature(friend_list, feature_list)
	# pd.set_option('display.max_rows', None)
	# pd.set_option('display.max_columns', None)
	# for i in list(person_feature_df):
	# 	print(person_feature_df[i])

	## add label 'circle'
	print("="*40)
	print("Label circle of Each person")
	person_feature_circle_df = circle_label(person_feature_df, circle_list)
	print(person_feature_circle_df)

	'''
	## Statistics
	print("="*40)
	print("Statistics")
	feature_counter = feature_count(person_feature_circle_df)
	print("Statistics Finish")
	feature_name = list(person_feature_circle_df)
	for idx, feat in enumerate(feature_counter):
		print(idx, feature_name[idx])
		for key, value in feat.items():
			print(key, value)
		print('-'*20)
	'''

	## adjacency_weight
	print("adjacency_weight")
	# print(len(feature_list))
	feature_list.remove('id')
	# print(len(feature_list))
	feature_weight = np.ones(len(feature_list))
	adjacency_weight = gen_adjacency_weight_v2(friend_list, feature_list, feature_weight, person_feature_df)
	print(adjacency_weight)
	print(len(friend_list), adjacency_weight.shape)
	# print(person_feature_df)
	# print(person_feature_df.at[friend_list[0],'birthday'])
	np.save(str(target)+'_adj', adjacency_weight)
	aaa = np.load(str(target)+'_adj.npy')
	print("aaa", aaa)



	