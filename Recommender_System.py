# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#dataset = pd.read_csv(r"E:\Swapnil\Data_Science\Machine_Learning\Recommendation_Engines\Collaborative_Filtering\USER_USER\USER_AE\User_AE_Rating_Map.csv")

dataset = pd.read_csv("E:/Swapnil/Data_Science/Machine_Learning/Recommendation_Engines/Collaborative_Filtering/USER_USER/USER_AE//User_AE_Rating_Map.csv")

# Creating the sparse matrix : For each user, for each group what is the rating score
# This will help in finding correlation between the Users based on rating provided to a perticular group
user_group_rating = dataset.pivot_table(index='GROUP_ID', columns='USER_ID', values='RATING_SCORE')

# To make sure correlation works properly, updating all the nan to 0
user_group_rating = user_group_rating.fillna(0)

"""
IDXXX00001_ratings = user_group_rating['IDXXX00001']
USERS_like_IDXXX00001 = pd.DataFrame(user_group_rating.corrwith(IDXXX00001_ratings), columns= ["CORRELATION"])
USERS_like_IDXXX00001['USER_ID_KEY'] = USERS_like_IDXXX00001.index
USERS_like_IDXXX00001 = USERS_like_IDXXX00001[USERS_like_IDXXX00001['CORRELATION']>0].reset_index()
USERS_like_IDXXX00001['SIMILAR_USERS_GROUP_KEY'] = 5
similar_users_df = similar_users_df.append(pd.DataFrame(USERS_like_IDXXX00001[['USER_ID_KEY', 'SIMILAR_USERS_GROUP_KEY']], columns=similar_users_df.columns))
similar_users_df = similar_users_df[['USER_ID_KEY', 'SIMILAR_USERS_GROUP_KEY']]
similar_users_df = similar_users_df.dropna(thresh=2)
"""

#similar_users_df = pd.DataFrame([])
similar_users_df = pd.DataFrame(columns=["USER_ID_KEY", "SIMILAR_USERS_GROUP_KEY"])
itr = 0
#for iterator_var in ['IDXXX00001', 'IDXXX00002', 'IDXXX00003', 'IDXXX00004', 'IDXXX00005']   :
for iterator_var in user_group_rating.columns:
    itr = itr+1
    print('itr',itr)
    ratings = user_group_rating[iterator_var]
    similar_users_all = pd.DataFrame(user_group_rating.corrwith(ratings), columns= ["CORRELATION"])
    #similar_users_all = user_group_rating.corrwith(ratings)
    similar_users_valid = similar_users_all[similar_users_all['CORRELATION']>0]
    similar_users_valid['USER_ID_KEY'] = similar_users_valid.index
    similar_users_valid['SIMILAR_USERS_GROUP_KEY'] = itr
    similar_users_df = similar_users_df.append(pd.DataFrame(similar_users_valid[['USER_ID_KEY', 'SIMILAR_USERS_GROUP_KEY']], columns=similar_users_df.columns))
    similar_users_df = similar_users_df[['USER_ID_KEY', 'SIMILAR_USERS_GROUP_KEY']]
# replace index with default index 
similar_users_df.reset_index(inplace = True, drop = True) 
