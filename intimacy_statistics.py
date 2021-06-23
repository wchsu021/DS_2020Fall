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

def degree_of_intimacy(key,circle_list,mutual_friend_list):
    n = len(circle_list[key])
    possible_friend_pairs = int(n*(n-1)/2)
    count_friend_pairs = 0
    for number1 in range(n-1):
        friend1 = circle_list[key][number1]
        for number2 in range(number1+1,n):
            friend2 = circle_list[key][number2]
            if friend2 in mutual_friend_list[friend1]:
                count_friend_pairs += 1
    doi = round(count_friend_pairs/possible_friend_pairs,2)
    
    return n,count_friend_pairs,possible_friend_pairs,doi

if __name__ == '__main__':


	target_list = [239,345,611,1357,1839,1968,2255,2365,2738,2790,2895,3059,3735,\
               4406,4829,5212,5494,5881,6413,6726,7667,8100,8239,8553,8777,\
                   9103,9642,9846,9947,10395,10929,11014,11186,11364,11410,\
                       12800,13353,13789,15672,16203,16378,16642,16869,17951,\
                           18005,18543,19129,19788,22650,22824,23157,23299,\
                               24758,24857,25159,25568,25773,26321,26492]

	df_intimacy = pd.DataFrame(columns=['Target','Circle','Circle Length', 'Friend Pairs','Possible Friend Pairs','Degree of Intimacy'])

	for target in target_list:
		ego_file = 'egonets/'+str(target)+'.egonet'
		circle_file = 'Training/'+str(target)+'.circles'
		friend_list, mutual_friend_list = read_egonet_file(ego_file)
		circle_list = read_circle_file(circle_file)
		feature_list = read_feature_list()
        
		for key, value in circle_list.items():
			n,count_friend_pairs,possible_friend_pairs,doi =\
			degree_of_intimacy(key,circle_list,mutual_friend_list)
            
			df_intimacy = df_intimacy.append({'Target':target,'Circle':key,\
            'Circle Length':n,'Friend Pairs':count_friend_pairs,\
             'Possible Friend Pairs':possible_friend_pairs,'Degree of Intimacy':doi},ignore_index=True)
	
	print(df_intimacy)
	df_intimacy.to_csv('Intimacy.csv')