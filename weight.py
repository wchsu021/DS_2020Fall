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

if __name__ == '__main__':
	
	target = 239
	ego_file = 'egonets/'+str(target)+'.egonet'
	
	## friend_list
	friend_list, mutual_friend_list = read_egonet_file(ego_file)

	## feature list
	feature_list = read_feature_list()

	## feature dataFrame
	person_feature_df = read_person_feature(friend_list, feature_list)

	## adjacency_weight
	print("adjacency_weight")
	feature_list.remove('id')
	
	feature_weight = np.ones(len(feature_list))
	adjacency_weight = gen_adjacency_weight(friend_list, feature_list, feature_weight, person_feature_df)
	
	print(len(friend_list), adjacency_weight.shape)
	