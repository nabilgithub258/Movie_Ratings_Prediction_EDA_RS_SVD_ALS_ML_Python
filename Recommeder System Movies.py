#!/usr/bin/env python
# coding: utf-8

# In[537]:


#####################################################################################################
######################### MOVIES DATA SET  ##########################################################
#####################################################################################################


# In[538]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[539]:


column_names = ['user_id','item_id','rating','timestamp']                     #### our data is not structured


# In[540]:


df = pd.read_csv('u.data',sep='\t',names=column_names)


# In[541]:


df.head()                                #### now it makes more sence with cols names


# In[542]:


movie_df = pd.read_csv('Movie_Id_Titles')


# In[543]:


movie_df.head()                          #### primary key is item_id from both tables


# In[544]:


df = pd.merge(df,movie_df,on='item_id')


# In[545]:


df.head()                             #### merged both dataframe into one with the help of primary key


# In[546]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[547]:


df[df.duplicated()]                         #### data is clean


# In[548]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[549]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[550]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[551]:


df.isnull().any()                          #### no missing values either


# In[552]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[553]:


df.head()


# In[554]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()               #### but this can be misleading


# In[555]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()               #### here we see more clear picture


# In[556]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())             #### making a dataframe as ratings to see the mean


# In[557]:


ratings.head()


# In[558]:


ratings['number_of_ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())         #### this will be more helpful to see how many people reviewed each movie


# In[559]:


ratings.head()                  #### much better and easy to understand


# In[560]:


######################################################################
############## Part V - EDA
######################################################################


# In[561]:


ratings.head()


# In[562]:


custom = {0:'purple',
         1:'red'}

g = sns.jointplot(x=ratings.rating,y=ratings.number_of_ratings,data=ratings,color='black')

g.fig.set_size_inches(17,9)


#### seems like as the ratings go up the number of people who reviewed also goes up except some obvious outliers


# In[563]:


ratings[ratings.rating == 5]                #### all these movies been given 5 star ratings but 
                                            #### theres only one person who reviewed it so this will throw off our recommendation system


# In[564]:


ratings.number_of_ratings.sort_values(ascending=False)             #### this is more realistic approach


# In[565]:


g = sns.jointplot(x='rating',y='number_of_ratings',data=ratings,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)

#### definately we do some correlation here


# In[566]:


######################################################################
############## Part VI - BASIC RECOMMENDATION SYSTEMS
######################################################################


# In[567]:


from scipy.stats import pearsonr                  #### lets see this with pearsonr


# In[568]:


co_eff, p_value = pearsonr(ratings.rating,ratings.number_of_ratings)


# In[569]:


co_eff


# In[570]:


p_value                         #### we reject null hypothesis


# In[571]:


df.groupby(['user_id','title'])['rating'].sum().unstack().fillna(0)     #### this way we can see people who reviewed one more and gave rating, did they give any other movies similar ratings


# In[572]:


starwars_user_rating = df.groupby(['user_id','title'])['rating'].sum().unstack()['Star Wars (1977)']

starwars_user_rating                    #### user ratings only related to star wars 1977 movie


# In[573]:


df.head()


# In[574]:


ratings.sort_values('number_of_ratings',ascending=False).head()             #### seems like star wars is the most reviewed movie and it holding above 4 across is quite marvellous


# In[575]:


ratings.number_of_ratings.mean()                    #### mean of the ratings given by the users, meaning the density lies within 60


# In[576]:


ratings.number_of_ratings.std()


# In[577]:


ratings.number_of_ratings.quantile(0.90)


# In[578]:


df.groupby(['user_id','title'])['rating'].sum().unstack().corrwith(starwars_user_rating)


# In[579]:


corr_starwars = df.groupby(['user_id','title'])['rating'].sum().unstack().corrwith(starwars_user_rating)


# In[580]:


corr_starwars.head()                     #### now this will give the recommendation based on the movie star wars 1977


# In[581]:


corr_starwars = pd.DataFrame(corr_starwars,columns=['Correlation'])


# In[582]:


corr_starwars.head()


# In[583]:


corr_starwars.info()


# In[584]:


corr_starwars.isnull().any()


# In[585]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[586]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr_starwars.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')            #### it makes sense because not every user will see all the movies and rate them


#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[587]:


corr_starwars.dropna(inplace=True)


# In[588]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr_starwars.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')


#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[589]:


corr_starwars.isnull().any()


# In[590]:


corr_starwars.info()


# In[591]:


corr_starwars.sort_values('Correlation',ascending=False)            #### this is a problem because obviously such movies are not correlated to star wars but they getting correlation 1.0


# In[592]:


corr_starwars = corr_starwars.join(ratings.number_of_ratings)


# In[593]:


corr_starwars.head()            #### now we know how many people reviewed


# In[594]:


corr_starwars = corr_starwars[corr_starwars.number_of_ratings > 100].sort_values('number_of_ratings',ascending=False)

corr_starwars

#### we have opted out of number of rating below 100 so our correlation can work properly which is now working much efficiently


# In[595]:


corr_starwars.sort_values('Correlation',ascending=False)              #### now we are seeing proper recommendation system


# In[596]:


heat = corr_starwars.sort_values('Correlation',ascending=False)


# In[597]:


fig, ax = plt.subplots(figsize=(40,22))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis')


# In[598]:


heat = corr_starwars.sort_values('Correlation',ascending=False).head(20)         #### top 20 highly correlated movies to star wars 1977


# In[599]:


fig, ax = plt.subplots(figsize=(20,12))

sns.heatmap(heat,ax=ax,linewidths=0.5,cmap='viridis',annot=True)


# In[600]:


heat['number_of_ratings'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black',markersize=15,linestyle='dashed',linewidth=4)

plt.title('Movies Rating Graph')

plt.xlabel('Movies')

plt.ylabel('Density')

#### much better to understand


# In[601]:


g = sns.lmplot(x='Correlation',y='number_of_ratings',data=heat,height=7,aspect=2,line_kws={'color':'red'},scatter_kws={'color':'black'})

plt.savefig('RS_correlation_number_ratings_lmplot.jpeg', dpi=300, bbox_inches='tight')

#### we see clear correlation, not suprised


# In[484]:


#### now we will do something quite interesting, EXCITED
#### we will make a function to do all the steps in one go and when we put movie name, 
#### it reflects the recommendation based on that movie

ratings.head()


# In[485]:


df.head()


# In[486]:


movie_matrix = df.pivot_table(index='user_id', columns='title', values='rating')


# In[487]:


movie_matrix.head()


# In[488]:


def find_similar_movies(movie_name, min_ratings=100):
    movie_ratings = movie_matrix[movie_name]
    similar_movies = movie_matrix.corrwith(movie_ratings)
    corr_movie = pd.DataFrame(similar_movies, columns=['Correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie = corr_movie.join(ratings['number_of_ratings'])
    recommendations = corr_movie[corr_movie['number_of_ratings'] > min_ratings].sort_values('Correlation', ascending=False)
    return recommendations


# In[489]:


find_similar_movies('Star Wars (1977)').head(10)                  #### please ignore the warning, its happening on correlation phase when its trying to divide by zero, but it has no impact on the result


# In[490]:


find_similar_movies('L.A. Confidential (1997)').head(10)


# In[491]:


find_similar_movies('Austin Powers: International Man of Mystery (1997)').head(10)


# In[492]:


find_similar_movies('Gone with the Wind (1939)').head(10)                #### one last one before we wrap this up


# In[493]:


############################################################################################################################
#### In summary, we have explored the fundamentals of building a movie recommendation system using Collaborative ###########
#### Filtering (CF) with a focus on correlation-based recommendations. By leveraging user ratings data, we can compute #####
#### similarities between movies to recommend those that align closely with a user's preferences. Our implementation #######
#### involved cleaning the dataset, creating a pivot table, calculating correlations, and filtering results to ensure ######
#### reliable recommendations. This approach, demonstrated through the example of recommending movies similar to ###########
#### "Star Wars (1977)", highlights how CF can provide personalized suggestions. ###########################################
############################################################################################################################


# In[494]:


######################################################################
############## Part VII - ADVANCED RECOMMENDATION SYSTEMS
######################################################################


# In[495]:


df.head()


# In[496]:


df.drop(columns='timestamp',inplace=True)

df.head()


# In[497]:


user_item_matrix = df.pivot_table(index='user_id', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

user_item_matrix                             #### movie matrix without NULL


# In[498]:


user_ratings_mean = np.mean(user_item_matrix, axis=1)
user_item_matrix_normalized = user_item_matrix - user_ratings_mean.values.reshape(-1, 1)

user_ratings_mean                        #### user ratings mean matrix


# In[499]:


user_item_matrix_normalized                   #### normalized user rating


# In[500]:


from scipy.sparse.linalg import svds


# In[501]:


U, sigma, Vt = svds(user_item_matrix_normalized, k=min(user_item_matrix_normalized.shape)-1)
sigma = np.diag(sigma)

sigma                        #### setting up SVD, we are basically breaking down whole normalized matrix into 3 components as U * sigma * Vt = original normalized matrix


# In[502]:


predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_item_matrix.columns)

predicted_ratings_df                 #### so basically this is our predicted ratings for all the user

#### you will see the predicted ratings are pretty much on point because our model basically had the leak data meaning we didn't split anything to test our predictions
#### next we will do the classic spliting and testing and then comparing the RMSE to see how well our predictions were


# In[503]:


user_item_matrix.head()                 #### this is the actual ratings


# In[505]:


user_id = 2
user_idx = user_id - 1  # user_id is 1-indexed because python indexes from 0

predicted_ratings_df.iloc[user_idx].head()                #### we are doing the predicted ratings for user 2


# In[506]:


predicted = predicted_ratings_df.values.flatten()
actual = user_item_matrix.values.flatten()

actual                         #### preparing for RMSE


# In[507]:


predicted


# In[508]:


mask = actual != 0
rmse = np.sqrt(mean_squared_error(predicted[mask], actual[mask]))

rmse               #### just like we predicted as the data is leaked so it doesn't suprise us why RMSE is so small here


# In[510]:


import implicit
from sklearn.model_selection import train_test_split


# In[518]:


df = df.drop_duplicates(subset=['user_id', 'title'])


# In[519]:


train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)


# In[520]:


train_user_item_matrix = train_data.pivot(index='user_id', columns='title', values='rating').fillna(0)
test_user_item_matrix = test_data.pivot(index='user_id', columns='title', values='rating').fillna(0)

#### giving same treatment for both test and train data


# In[521]:


user_ratings_mean = np.mean(train_user_item_matrix, axis=1)
train_user_item_matrix_normalized = train_user_item_matrix - user_ratings_mean.values.reshape(-1, 1)

#### normalizing the data for train


# In[522]:


U, sigma, Vt = svds(train_user_item_matrix_normalized, k=50)
sigma = np.diag(sigma)

#### applying SVD


# In[523]:


pred = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=train_user_item_matrix.columns)

predicted_ratings_df.head()               #### using SVD and normalizing to get the predicted df


# In[527]:


test_user_item_matrix_aligned = test_user_item_matrix.reindex(index=predicted_ratings_df.index, columns=predicted_ratings_df.columns).fillna(0)
rmse = np.sqrt(mean_squared_error(test_user_item_matrix_aligned.values.flatten(), predicted_ratings_df.values.flatten()))

rmse                        #### not bad knowing our ratings go from 1-5


# In[529]:


from scipy.sparse import coo_matrix                       #### lets see more advanced method like ALS to reduce RMSE


# In[528]:


sparse_train_matrix = coo_matrix(train_user_item_matrix.values)

sparse_train_matrix                  #### we will be doing ALS so we need it in this format


# In[530]:


from implicit.als import AlternatingLeastSquares


# In[531]:


als_model = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=20)


# In[532]:


als_model.fit(sparse_train_matrix.T)


# In[533]:


user_factors = als_model.user_factors
item_factors = als_model.item_factors


# In[534]:


predicted_ratings = np.dot(user_factors, item_factors.T).T

#### transposing the factors


# In[535]:


predicted_ratings_df = pd.DataFrame(predicted_ratings, index=train_user_item_matrix.index, columns=train_user_item_matrix.columns)

#### making into a dataframe


# In[536]:


test_user_item_matrix_aligned = test_user_item_matrix.reindex(index=predicted_ratings_df.index, columns=predicted_ratings_df.columns).fillna(0)
rmse = np.sqrt(mean_squared_error(test_user_item_matrix_aligned.values.flatten(), predicted_ratings_df.values.flatten()))

rmse             #### we did reduce RMSE from 0.69 to 0.51 which is amazing and massive improvement


# In[ ]:


############################################################################################################################
#### In our recommendation system project, we have achieved significant results through a methodical approach. #############
#### Initially, we implemented Singular Value Decomposition (SVD) and obtained an RMSE of 0.69. Subsequently, we ###########
#### improved our model using Alternating Least Squares (ALS), achieving a further reduced RMSE of 0.50.While we ###########
#### considered exploring more advanced methods such as neural networks, we have decided to conclude our work at this ######
#### stage, satisfied with the improvements and insights gained from ALS. ##################################################
############################################################################################################################

