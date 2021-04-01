# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 15:37:46 2021

@author: USER
"""


import numpy as np
import pandas as pd

MIN_TARGET_FREQ = 4 # drop target/city_id values that appear less than this many times, as final step's target 
KEEP_TOP_K_TARGETS = 0 # keep K most frequent city ID targets (redundnat with the above, )

## (some) categorical variables that appear less than this many times will be replaced with a placeholder value!
## Includes CITY id (but done after target filtering, to avoid creating a "rare class" target:
LOW_COUNT_THRESH = 10

RUN_TABNET = False
max_epochs = 20

GET_COUNT_AGG_FEATS = False ## disable getting count, rank etc' groupby features , for speedup

DROP_FIRST_ROW =  True #False  ## drop first interaction per user from train data

## for matrix factorization/CF:
### morte possible ID_cols :  # last (last step in trip) - would double data per user incorrectly
### hotel_country_lag1 , city_id_lag1  (very relevant - needs shared embeddingm and would increase cardinality a lot.. ) 
ID_COLS = [
#     'device_class',
#            'affiliate_id',
           'booker_country',
           'checkin_quarter',
#            "last",
          "first_hotel_country"] 
MF_KEEP_COLS = ["ID"]+ID_COLS+['city_id',"hotel_country"]

SAVE_TO_DISK = False

TARGET_COL =  'city_id' #"city_id"#'hotel_country' 
USER_ID_COL = "utrip_id"

BASE_CAT_COLS = ['city_id',  'affiliate_id', 'booker_country', 'hotel_country']


# https://stackoverflow.com/questions/33907537/groupby-and-lag-all-columns-of-a-dataframe
# https://stackoverflow.com/questions/62924987/lag-multiple-variables-grouped-by-columns
## lag features with groupby over many columns: 
def groupbyLagFeatures(df:pd.DataFrame,lag:[]=[1,2],group="utrip_id",lag_feature_cols=[]):
    """
    lag features with groupby over many columns.
    Assumes sorted data!
    https://stackoverflow.com/questions/62924987/lag-multiple-variables-grouped-by-columns"""
    if len(lag_feature_cols)>0:
        df=pd.concat([df]+[df.groupby(group)[lag_feature_cols].shift(x).add_prefix('lag'+str(x)+"_") for x in lag],axis=1)
    else:
         df=pd.concat([df]+[df.groupby(group).shift(x).add_prefix('lag'+str(x)+"_") for x in lag],axis=1)
    return df


def groupbyFirstLagFeatures(df:pd.DataFrame,group="user_id",lag_feature_cols=[]):
    """
    Get  first/head value lag-like of features with groupby over columns. Assumes sorted data!
    """
    if len(lag_feature_cols)>0:
        df=pd.concat([df]+[df.groupby(group)[lag_feature_cols].transform("first").add_prefix("first_")],axis=1)
    else:
#          df=pd.concat([df]+[df.groupby(group).first().add_prefix("first_")],axis=1)
        df=pd.concat([df]+[df.groupby(group).transform("first").add_prefix("first_")],axis=1)
    return df

######## Get n most popular items, per group
def most_popular(group, n_max=4):
    """Find most popular hotel clusters by destination
    Define a function to get most popular hotels for a destination group.

    Previous version used nlargest() Series method to get indices of largest elements. But the method is rather slow.
    Source: https://www.kaggle.com/dvasyukova/predict-hotel-type-with-pandas
    """
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    most_popular = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return np.array_str(most_popular)[1:-1] # remove square brackets


## https://codereview.stackexchange.com/questions/149306/select-the-n-most-frequent-items-from-a-pandas-groupby-dataframe
# https://stackoverflow.com/questions/52073054/group-by-a-column-to-find-the-most-frequent-value-in-another-column
## can get modes (sorted)
# https://stackoverflow.com/questions/50592762/finding-most-common-values-with-pandas-groupby-and-value-counts
## df.groupby('tag')['category'].agg(lambda x: x.value_counts().index[0])
# https://stackoverflow.com/questions/15222754/groupby-pandas-dataframe-and-select-most-common-value
# source2.groupby(['Country','City'])['Short name'].agg(pd.Series.mode)





df = pd.read_csv('/root/booking_train_set.csv',
#                  nrows=523456,
                 index_col=[0],
                 parse_dates=["checkin","checkout"],infer_datetime_format=True)

df.sort_values([ "checkin",
                "user_id"],inplace=True)
print(df.nunique())
df["checkin_week"] = df["checkin"].dt.isocalendar().week.astype(int) ## week of year
df["checkin_month"] = df["checkin"].dt.month

df["checkin_quarter"] = df["checkin"].dt.quarter

obj_cols_list = ['device_class','booker_country','hotel_country',
#                 "city_id"
                ] # we could also define when loading data, dtype


df["utrip_steps_from_end"] = df.groupby("utrip_id")["checkin"].rank(ascending=True,pct=True) 

df["row_num"] = df.groupby("utrip_id")["checkin"].rank(ascending=True,pct=False).astype(int)
utrip_counts = df["utrip_id"].value_counts()
df["total_rows"] = df["utrip_id"].map(utrip_counts)

### last step in trip
df["last"] = (df["total_rows"]==df["row_num"]).astype(int)

df["total_rows"].describe();

affiliates_counts = df["affiliate_id"].value_counts()
print("before:", affiliates_counts)
print("uniques",df["affiliate_id"].nunique())
affiliates_counts = affiliates_counts.to_dict()
# df["affiliate_id"] = df["affiliate_id"].where(df["affiliate_id"].apply(lambda x: x.map(x.value_counts()))>=3, -1)
df["affiliate_id"] = df["affiliate_id"].where(df["affiliate_id"].map(affiliates_counts)>=3, -2)
df["affiliate_id"] = df["affiliate_id"].astype(int)

print("after\n",df["affiliate_id"].value_counts())
print("uniques",df["affiliate_id"].nunique())

df = groupbyFirstLagFeatures(df,group=['device_class', 'affiliate_id',
                                       'booker_country','checkin_month',"last"],
                             lag_feature_cols=["hotel_country","city_id",'device_class', 'affiliate_id',
                                              "checkin_quarter","checkin_month","booker_country"])


print(df[["first_hotel_country","hotel_country","city_id","first_city_id"]].nunique())

print(df[ID_COLS + ['checkin_month',"total_rows"]].nunique(axis=0))
df.groupby(ID_COLS).size()

df["ID"] = df[ID_COLS].astype(str).sum(1)#.astype("category")

df_ID = df[["ID"]+ID_COLS].drop_duplicates().set_index("ID")
df_ID

print(df["ID"].value_counts().describe())
print(df["ID"].tail(3))
print(df["ID"].nunique())


if GET_COUNT_AGG_FEATS:
    count_cols = [ 'city_id','affiliate_id', 'booker_country', 'hotel_country', 
    #               'utrip_id','user_id', 
     "checkin_month","checkin_week"]
    for c in count_cols:
        df[f"{c}_count"] = df.groupby([c])["duration"].transform("size")

    ########################################################
    ## nunique per trip
    ### https://stackoverflow.com/questions/46470743/how-to-efficiently-compute-a-rolling-unique-count-in-a-pandas-time-series

    nunique_cols = [ 'city_id','affiliate_id', 'booker_country', 'hotel_country']
    # df["nunique_booker_countries"] = df.groupby("utrip_id")["booker_country"].nunique()
    # df["nunique_hotel_country"] = df.groupby("utrip_id")["hotel_country"].nunique()
    for c in nunique_cols:
        df[f"{c}_nunique"] = df.groupby(["utrip_id"])[c].transform("nunique")
    print(df.nunique())

    ########################################################
    ## get frequency/count feature's rank within a group - e.g. within a country (or affiliate) 
    ## add "_count" to column name to get count col name, then add rank col 

    ### ALT/ duplicate feat - add percent rank (instead or in addition)

    rank_cols = ['city_id','affiliate_id', 'booker_country','hotel_country',
     "checkin_month"]
    ### what is meaning of groupby and rank of smae variable by same var? Surely should be 1 / unary? 
    for c in rank_cols:
        df[f"{c}_rank_by_hotel_country"] = df.groupby(['hotel_country'])[f"{c}_count"].transform("rank")
        df[f"{c}_rank_by_booker_country"] = df.groupby(['booker_country'])[f"{c}_count"].transform("rank")
        df[f"{c}_rank_by_affiliate"] = df.groupby(['affiliate_id'])[f"{c}_count"].transform("rank")     
else:
    freq = df["city_id"].value_counts()
    df["city_id_count"] = df["city_id"].map(freq)
    print(freq.describe())
    
    
print("cities with more than 7 occurences:",df.loc[df["city_id_count"]>=7]["city_id"].nunique())
print(f"cities with more than MIN_TARGET_FREQ ({MIN_TARGET_FREQ}) occurences:",df.loc[df["city_id_count"]>=MIN_TARGET_FREQ]["city_id"].nunique())
print(f"rows left if filtering by MIN_TARGET_FREQ :",df.loc[df["city_id_count"]>=MIN_TARGET_FREQ].shape[0])

if KEEP_TOP_K_TARGETS > 0 :
    df_end = df.loc[df["utrip_steps_from_end"]==1].drop_duplicates(subset=["city_id","hotel_country","user_id"])[["city_id","hotel_country"]].copy()
    print(df_end.shape[0])
    end_city_counts = df_end.city_id.value_counts()
    print(end_city_counts)
    
    TOP_TARGETS = end_city_counts.head(KEEP_TOP_K_TARGETS).index.values
    print(f"top {KEEP_TOP_K_TARGETS} targets \n",TOP_TARGETS)

### unsure about this filtering - depends if data points are real or mistake
print("dropping users with less than 4 trips")
df2 = df.loc[df["total_rows"]>=4]#.copy()
# print("abnormal users dropped",df.shape[0]-df2.shape[0])

print(f"dropping cities with less than {MIN_TARGET_FREQ} occurences:")
df2 = df2.loc[df2.groupby(["city_id"])["hotel_country"].transform("count")>=MIN_TARGET_FREQ] ## update count
# df2 = df2.loc[df2["city_id_count"]>=MIN_TARGET_FREQ]
print(df2.shape[0])

# print(f"dropping users with less than 4 instances, after previous city filter:")
df2 = df2.loc[df2.groupby(["utrip_id"])[TARGET_COL].transform("count")>=4]

# print(f"dropping cities with less than {MIN_TARGET_FREQ} occurences:")
df2 = df2.loc[df2.groupby(["city_id"])["hotel_country"].transform("count")>=MIN_TARGET_FREQ]

print(f"dropping users with less than 4 instances, after previous filters:")
df2 = df2.loc[df2.groupby(["utrip_id"])[TARGET_COL].transform("count")>=4]

df2 = df2.loc[df2.groupby(["city_id"])["hotel_country"].transform("count")>=MIN_TARGET_FREQ]
df2 = df2.loc[df2.groupby(["utrip_id"])[TARGET_COL].transform("count")>=4]
df2 = df2.loc[df2.groupby(["city_id"])["hotel_country"].transform("count")>=MIN_TARGET_FREQ]
df2 = df2.loc[df2.groupby(["utrip_id"])[TARGET_COL].transform("count")>=4]
df2 = df2.loc[df2.groupby(["city_id"])["hotel_country"].transform("count")>=MIN_TARGET_FREQ]
df2 = df2.loc[df2.groupby(["utrip_id"])[TARGET_COL].transform("count")>=4]

print("rows left:",df2.shape[0])

print("nunique cities after freq filt",df2[["city_id","utrip_id","user_id","hotel_country"]].nunique())
print("nunique city_id per hotel_country:")
df2.groupby(["hotel_country"])["city_id"].nunique().describe()


df = df2

def sample_hyperparameters():
    """
    Yield possible hyperparameter choices.
    """

    while True:
        yield {
            "no_components": np.random.randint(16, 64),
#             "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-7), # was 1e-08
            "user_alpha": np.random.exponential(1e-7),
#             "max_sampled": np.random.randint(5, 40),
#             "num_epochs": np.random.randint(1, 8),
            "num_epochs":np.random.randint(5, 20)
        }
        
        
def random_search(train, test,item_features=None,user_features=None, num_samples=10, num_threads=16):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.

    Parameters
    ----------

    train: np.float32 coo_matrix of shape [n_users, n_items]
        Training data.
    test: np.float32 coo_matrix of shape [n_users, n_items]
        Test data.
    num_samples: int, optional
        Number of hyperparameter choices to evaluate.


    Returns
    -------

    generator of (auc_score, hyperparameter dict, fitted model)

    """

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")
        model = LightFM(**hyperparams) # ,learning_rate=.03
        model.fit(train, epochs=num_epochs, num_threads=num_threads,
                 item_features=item_features, user_features=user_features)
        ### should i pass in train_interactions (when i have repeats) ? 
#         score = auc_score(model, test, train_interactions=train, num_threads=num_threads).mean() # ORIG
#         score = precision_at_k(model, test,  num_threads=num_threads,k=4).mean()
        score = recall_at_k(model, test,  num_threads=num_threads,k=90
                           ,item_features=item_features,user_features=user_features).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)


def create_feature_list(df_,cols):
    feature_list = []
    for col in cols:
        feature_list += list(set(df_[col].to_list()))
    return feature_list
  

def build_features(df_):
    return [(row[0],
             [row[i] for i in range(1,len(row))]) 
            for row in df_.itertuples(index=False)]


from datetime import datetime, timedelta
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix 
from sklearn.metrics import roc_auc_score
import time
from lightfm.evaluation import auc_score

from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from lightfm.cross_validation import random_train_test_split 
import itertools
import multiprocessing as mp


dataset = Dataset()

# dataset.fit(df[USER_ID_COL].unique(), df[TARGET_COL].unique())
dataset.fit(df[USER_ID_COL], df[TARGET_COL])

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

interactions_matrix, weights_matrix = dataset.build_interactions([tuple(i) for i in df[[USER_ID_COL,TARGET_COL]].values])

(train, test) = random_train_test_split(interactions=interactions_matrix, test_percentage=0.2)

(score, hyperparams, model) = max(random_search(train, test, num_threads=16, num_samples=25), key=lambda x: x[0])
print("Best score {} at {}".format(score, hyperparams))
#MIN_TARGET_FREQ=3
#p@4 Best score 0.5240530753196541 at {'no_components': 45, 'loss': 'warp', 'learning_rate': 0.05558676440957299, 'item_alpha': 2.194976636780396e-07, 'user_alpha': 4.9008757699557144e-08, 'num_epochs': 20}
#P@90 Best score 0.8167861293520302 at {'no_components': 56, 'loss': 'warp', 'learning_rate': 0.04062078353812153, 'item_alpha': 1.133949772432684e-07, 'user_alpha': 1.5952753120840533e-07, 'num_epochs': 18}
#p@90 MIN_TARGET_FREQ = 20 Best score 0.8962797203678623 at {'no_components': 49, 'loss': 'warp', 'learning_rate': 0.018680248667683256, 'item_alpha': 2.0040298117787927e-08, 'user_alpha': 6.048391712979686e-08, 'num_epochs': 16}
####
###Last run
#P@90 Best score 0.8279699521989476 at {'no_components': 63, 
#'loss': 'warp', 'learning_rate': 0.03455394715240906, 'item_alpha': 5.80963131317218e-08, 'user_alpha': 3.024859188656009e-08}
"""
{'no_components': 63,
 'loss': 'warp',
 'learning_rate': 0.03455394715240906,
 'item_alpha': 5.80963131317218e-08,
 'user_alpha': 3.024859188656009e-08}
"""


num_epochs = hyperparams.pop("num_epochs")
model = LightFM(**hyperparams) # ,learning_rate=.03
model.fit(interactions_matrix, epochs=num_epochs, num_threads=16)


def chunks(l_users, n):
        """Yield n number of striped chunks from l."""
        for i in range(0, n):
            yield l_users[i::n]

def worker_func(model,users,user_map,items_map, k=100):
    items_map = {v:k for k,v in items_map.items()}
    user_map = {v:k for k,v in user_map.items()}
    litems = list(items_map.keys()) 
    
    for u in users:
        scores = model.predict(u, litems)
        sdict[user_map[u]] = [items_map[i] for i in np.argsort(-scores)[:k]]    
    

def predict_mp(model,n_users,user_map,items_map,n_threads=mp.cpu_count()):
    
    
    processes = [mp.Process(target=worker_func,
                                         args=(model,d,user_map,items_map))
                 for d in chunks(range(n_users), n_threads)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


            
def predict_df(model,users,user_map,items_map, k=100):
    items_map = {v:k for k,v in items_map.items()}
    user_map = {v:k for k,v in user_map.items()}
    litems = list(items_map.keys())
    all_df = pd.DataFrame()
    #sdict = {}
    for u in users:
        #temp_df = pd.DataFrame(user_map[u]*k)
        scores = model.predict(u, litems)
        #sdict[user_map[u]] = [items_map[i] for i in np.argsort(-scores)[:k]]
        temp_df = pd.DataFrame({'user_id':[user_map[u]]*k,'recom':[items_map[i] for i in np.argsort(-scores)[:k]]})
        all_df = pd.concat([all_df,temp_df],ignore_index=True)
    return all_df


#recom = predict(model,range(num_users),dataset.mapping()[0],dataset.mapping()[2])
manager = mp.Manager()
sdict = manager.dict()
predict_mp(model, num_users,dataset.mapping()[0],dataset.mapping()[2])
recom_df = pd.DataFrame(dict(sdict).items(),
                                columns=['user_id','Recom'])
recom_df = recom_df.explode('Recom').reset_index(drop=True)
recom_df.to_csv('train_predictions.csv')




"""

df_item_features = df[["city_id","hotel_country"]].drop_duplicates()
features_list = create_feature_list(df_item_features,cols=["hotel_country"])

fdataset = Dataset()

# dataset.fit(df[USER_ID_COL].unique(), df[TARGET_COL].unique())
dataset.fit(df[USER_ID_COL], df[TARGET_COL],item_features=features_list)

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

interactions_matrix, weights_matrix = dataset.build_interactions([tuple(i) for i in df[[USER_ID_COL,TARGET_COL]].values])

(train, test) = random_train_test_split(interactions=interactions_matrix, test_percentage=0.2)

item_features = dataset.build_item_features(build_features(df_item_features))

(score1, hyperparams1, model1) = max(random_search(train, test,item_features=item_features, num_threads=16, num_samples=25), key=lambda x: x[0])
print("Best score {} at {}".format(score1, hyperparams1))
#MIN_TARGET_FREQ=3
#p@4 Best score 0.4536913405876642 at {'no_components': 47, 'loss': 'warp', 'learning_rate': 0.08757457890186483, 'item_alpha': 1.8861083850907907e-07, 'user_alpha': 8.781473211056553e-08, 'num_epochs': 20}
#p@90 Best score 0.7578771459284159 at {'no_components': 53, 'loss': 'warp', 'learning_rate': 0.07281070048728282, 'item_alpha': 1.1881631335778865e-08, 'user_alpha': 1.827456635866037e-07, 'num_epochs': 11}
"""