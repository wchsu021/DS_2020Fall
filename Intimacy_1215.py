import pandas as pd
import numpy as np
from scipy import stats

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

def feature_filter(feature_counter,feature_name,N):
    repeat = []
    repeat_feature = []

    for idx, feat in enumerate(feature_counter):
        #print(idx, feature_name[idx])        
        count = [0]
        for key, value in feat.items():
            #print(key,value)            
            count.append(value)
        #print(count)
        #print('-'*20)
        #print(count)  
        if max(count) > N:
            repeat.append([feature_name[idx],max(count)])
            repeat_feature.append(feature_name[idx])
            
    print('Feature Repeated more than '+str(int(N)))
    for item in repeat:
            print(item)    
    
    return repeat_feature
    
def chi2_test(table,obj_count):
    row_total = obj_count
    total_number = sum(row_total)
    column_total = sum(table)
    expected_table = np.array([row_total[0]*column_total/total_number])
    expected_table = np.append(expected_table,[row_total[1]*column_total/total_number],axis=0)
    #print(expected_table)
    
    chi_square = 0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            chi_square = chi_square + (table[i][j]-expected_table[i][j])**2/expected_table[i][j]
   
    dof = (table.shape[0]-1)*(table.shape[1]-1)
    p_value = 1-stats.chi2.cdf(chi_square,dof)
    
    alpha = 0.05
    
    #print(chi_square)
    #print(p_value)
    return p_value < alpha

def chi2_test_for_a_given_feature(key,feature_test):
    in_circle_df = person_feature_circle_df.loc[circle_list[key],repeat_feature_1]
    out_circle_friend_list = [item for item in friend_list if item not in circle_list[key]]
    out_circle_df = person_feature_circle_df.loc[out_circle_friend_list,repeat_feature_1]
    
    x = in_circle_df[feature_test]
    y = out_circle_df[feature_test]
    
    data = np.zeros([3,len(x)+len(y)])
    data[0] = data[0]-1
    for item in x:
        if item != None:
            #print(item)
            if ',' in item:
                item_list = item.split(',')
            else:
                item_list = [item]
            for d in item_list:
                if int(d) in data[0,:]:
                    idx = np.where(data[0]==int(d))[0][0]
                    data[1][idx] += 1
                else:
                    idx = np.where(data[0]==-1)[0][0]
                    data[0][idx] = d
                    data[1][idx] = 1
    
    for item in y:
        if item != None:
            #print(item)
            if ',' in item:
                item_list = item.split(',')
            else:
                item_list = [item]
            for d in item_list:
                if int(d) in data[0,:]:
                    idx = np.where(data[0]==int(d))[0][0]
                    data[2][idx] += 1
                else:
                    idx = np.where(data[0]==-1)[0][0]
                    data[0][idx] = d
                    data[2][idx] = 1
    col_num = len(data[0][data[0]!=-1])
    table = np.array([data[1][0:col_num],data[2][0:col_num]])
    obj_count = np.array([len(x),len(y)])
    #print(table)
    
    return chi2_test(table,obj_count)


#=============================
# Dearest

def how_many_circle_of_each_friend(target):
    ego_file = 'egonets/'+str(target)+'.egonet'
    circle_file = 'Training/'+str(target)+'.circles'
    friend_list, mutual_friend_list = read_egonet_file(ego_file)
    circle_list = read_circle_file(circle_file)
    circle_distribution = np.zeros([2,len(friend_list)])
    for k in range(len(friend_list)):
        pid = friend_list[k]
        circle_distribution[0][k] = pid
        circle_distribution[1][k] = len(search_circle(pid,circle_list))
       
    return circle_distribution

def max_circle_friend(circle_distribution):    
    max_circle = max(circle_distribution[1])
    index = [i for i, j in enumerate(circle_distribution[1]) if j == max_circle]
    most_circle_friend_tmp = circle_distribution[0][index]
    max_circle=int(max_circle)
    most_circle_friend = [int(item) for item in most_circle_friend_tmp]
    
    return most_circle_friend,max_circle

# Intimacy 

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


__name__ = '...'

if __name__ == '__main__':
	
    target = 1357
    ego_file = 'egonets/'+str(target)+'.egonet'
    circle_file = 'Training/'+str(target)+'.circles'
    friend_list, mutual_friend_list = read_egonet_file(ego_file)
    circle_list = read_circle_file(circle_file)
    feature_list = read_feature_list()
	
	## friend_list
    print("friend_list", len(friend_list))
    print(friend_list)
	
	## circle
    all_circle = []
    print("="*40)
    print("circle")
    for key, value in circle_list.items():
        all_circle.append(key)
        print(key, value)

	## feature dataFrame
    print("="*40)
    print("Feature of Each person")
    person_feature_df = read_person_feature(friend_list, feature_list)
    print(person_feature_df)

	## add label 'circle'
    print("="*40)
    print("Label circle of Each person")
    person_feature_circle_df = circle_label(person_feature_df, circle_list)
    print(person_feature_circle_df)

if __name__ == '__main__':

	## Statistics
    print("="*40)
    print("Statistics")
    feature_counter = feature_count(person_feature_circle_df)
    print("Statistics Finish")
    feature_name = list(person_feature_circle_df)

    print("="*40)
    print("Filter 1 Result:")
    # Use 1 as the least count of feature 
    
    repeat_feature_1 = feature_filter(feature_counter,feature_name,1)

    useless_feature = ['education;classes;from;name','education;classes;with;name',\
                       'education;degree;name','education;year;name',\
                        'work;from;name','work;location;name',\
                        'work;position;name','work;projects;from;name',\
                        'work;projects;with;name','work;with;name']   

    repeat_feature_1 = [item for item in repeat_feature_1 if item not in useless_feature]   

filter2 = False
if filter2 == True:

    print("="*40)
    print("Filter 2 Result:")
    # Use (circle's person number / 3) as the least count of feature
    for key in all_circle:
        df = person_feature_circle_df.loc[circle_list[key],repeat_feature_1]
        #print(df)
        
        feature_counter_2 = feature_count(df)
        feature_name_2 = list(df)
        print('Circle: '+key)
        print('Number:'+str(df.shape[0]))
        repeat_feature_2 = feature_filter(feature_counter_2,repeat_feature_1,df.shape[0]/3)
        print('='*40)

        df_2 = person_feature_circle_df.loc[circle_list[key],repeat_feature_2]
        print(df_2)

filter_chi2 = False
if filter_chi2 == True:  
    key = '87'
    feature_selected_by_chi2_for_a_cricle = []
    for feature_test in  repeat_feature_1:
        if chi2_test_for_a_given_feature(key,feature_test) == True:
            feature_selected_by_chi2_for_a_cricle.append(feature_test)          
    print(feature_selected_by_chi2_for_a_cricle)
    
    
filter_comparison = False
if filter_comparison == True: 
    print("="*40)
    print("Filter Compare Result:")
    # Use (circle's person number / 3) as the least count of feature
    for key in all_circle:
        df = person_feature_circle_df.loc[circle_list[key],repeat_feature_1]  
        feature_counter_2 = feature_count(df)
        feature_name_2 = list(df)
        print('Circle: '+key)
        print('Number:'+str(df.shape[0]))
        repeat_feature_2 = feature_filter(feature_counter_2,repeat_feature_1,df.shape[0]/3)
        print('-'*40)

        df_2 = person_feature_circle_df.loc[circle_list[key],repeat_feature_2]
        #print(df_2)
        print('Significant under chi2 test')
        feature_selected_by_chi2_for_a_cricle = []
        for feature_test in  repeat_feature_1:
            if chi2_test_for_a_given_feature(key,feature_test) == True:
                feature_selected_by_chi2_for_a_cricle.append(feature_test)          
        print(feature_selected_by_chi2_for_a_cricle)
        print('='*40)
    
    
    
# Dearest Friend  ============================================



df_dearest_friend = pd.DataFrame(columns=['Target','Dearest Friend','Number of Dearest Friend',\
                                'Dearest Friend Circle','Total Circle','Fraction'])


target_list = [239,345,611,1357,1839,1968,2255,2365,2738,2790,2895,3059,3735,\
               4406,4829,5212,5494,5881,6413,6726,7667,8100,8239,8553,8777,\
                   9103,9642,9846,9947,10395,10929,11014,11186,11364,11410,\
                       12800,13353,13789,15672,16203,16378,16642,16869,17951,\
                           18005,18543,19129,19788,22650,22824,23157,23299,\
                               24758,24857,25159,25568,25773,26321,26492]  

    
dearest = 0    
if dearest == True:        
    for target in target_list:
    
        ego_file = 'egonets/'+str(target)+'.egonet'
        circle_file = 'Training/'+str(target)+'.circles'
        friend_list, mutual_friend_list = read_egonet_file(ego_file)
        circle_list = read_circle_file(circle_file)
        
        total_circle = len(circle_list)
        circle_distribution = how_many_circle_of_each_friend(target)
        most_circle_friend,max_circle = max_circle_friend(circle_distribution)
        fraction = round(max_circle/total_circle,2)
        number_of_dearest_friend = len(most_circle_friend)
        
        df_dearest_friend = df_dearest_friend.append({'Target':target,'Dearest Friend':most_circle_friend,\
            'Dearest Friend Circle':max_circle,'Total Circle':total_circle,\
             'Number of Dearest Friend':number_of_dearest_friend,'Fraction':fraction},ignore_index=True)
    print(df_dearest_friend)
    
    df_dearest_friend.to_csv('Dearest Friend.csv')
    
    

df_intimacy = pd.DataFrame(columns=['Target','Circle','Circle Length',\
                                'Friend Pairs','Possible Friend Pairs','Degree of Intimacy'])

intimacy = True
if intimacy == True:
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







