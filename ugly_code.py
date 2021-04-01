
import collection
import pandas as pd
import pandas_gbq
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k, auc_score
import hnswlib
NUM_THREADS=16

def read_bq(q):
    return pandas_gbq.read_gbq(q,use_bqstorage_api=True)



def train_fm(train_,epochs=30,items_features= None,users_features=None,loss='warp',n_comp=64):
    model_ = LightFM(loss=loss,no_components=n_comp)
    model_.fit(train_, epochs=epochs,
              item_features=items_features,
              user_features=users_features, 
              num_threads=16,
              verbose=True)
    return model_


def evaluate_fm(model_,te_,tr_,
             items_features=None,
             users_features=None):
    if not tr_.multiply(te_).nnz == 0:
        print('train test interaction are not fully disjoin')

    # Compute and print the AUC score
    train_auc = auc_score(model_, tr_,
                          item_features=items_features,
                          user_features=users_features, 
                          num_threads=NUM_THREADS).mean()
    print('Collaborative filtering train AUC: %s' % train_auc)

    test_auc = auc_score(model_, te_, 
                         train_interactions=tr_,
                         item_features=items_features,
                         user_features=users_features, 
                         num_threads=NUM_THREADS).mean()
    print('Collaborative filtering test AUC: %s' % test_auc)
    p_at_k_train = precision_at_k(model_, tr_,
                                  item_features=items_features,
                                  user_features=users_features,
                                  k=5,num_threads=NUM_THREADS).mean()
    p_at_k_test = precision_at_k(model_, te_,train_interactions=tr_,
                                 item_features=items_features,
                                 user_features=users_features, 
                                 k=5,num_threads=NUM_THREADS).mean()

    print("Train precision: %.2f" % p_at_k_train)
    print("Test precision: %.2f" % p_at_k_test)


def create_feature_list(df_,cols):
    feature_list = []
    for col in cols:
        feature_list += list(set(df_[col].to_list()))
    return feature_list
  

def build_features(df_):
    return [(row[0],
             [row[i] for i in range(1,len(row))]) 
            for row in df_.itertuples(index=False)]


def sample_recommendation(model, train,items_labels,user_ids,item_features=None,user_features=None):

    #number of users and movies in training data
    n_users, n_items = train.shape

    #generate recommendations for each user we input
    for user_id in user_ids:

        #movies they already like
        #known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        known_positives = items_labels[train.tocsr()[user_id].indices]
        
        #movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items),user_features=user_features,item_features=item_features)
        #rank them in order of most liked to least
        top_items = items_labels[np.argsort(-scores)]

        #print out the results
        print("User %s" % user_id)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("        %s" % x)

        print("     Recommended:")

        for x in top_items[:3]:
            print("        %s" % x)


    
    
uid = pd.read_csv('for_dan.csv')

items = uid[['article_id','section_primary','writer_name']].drop_duplicates().reset_index(drop=True)
users = uid[['uid','popular_section','popular_platform','popular_sources']].drop_duplicates()

dataset = Dataset()
features_list = create_feature_list(items,cols=['section_primary','writer_name'])
user_features_list = create_feature_list(users,cols=['popular_section','popular_platform','popular_sources'])

#features_list = list(set(items.writer_name.to_list()))
dataset.fit(users=uid.uid.unique(),
            items=uid.article_id.unique(),
            item_features=features_list,
            user_features=user_features_list)

(interactions, weights) = dataset.build_interactions((x.uid,x.article_id) for x in uid.itertuples())
n_users, n_items = interactions.shape
1 - (interactions.getnnz() / (interactions.shape[0] * interactions.shape[1]))
item_features = dataset.build_item_features([(i.article_id,[i.section_primary,i.writer_name]) for i in items.itertuples()])
user_features = dataset.build_user_features([(u.uid,[u.popular_section]) for u in users.itertuples()])

item_features = dataset.build_item_features(build_features(items))
user_features = dataset.build_user_features(build_features(users))
user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()