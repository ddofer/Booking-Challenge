# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:53:43 2021

@author: USER
"""



import multiprocessing
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn import preprocessing



def split_df(df_, n_cores=multiprocessing.cpu_count()):
    """Spliting DataFrame into chunks"""

    batch_size = round(df_.shape[0]/n_cores)
    rest = df_.shape[0]%batch_size 
    cumulative = 0
    for i in range(n_cores):
        cumulative += batch_size
        if i == n_cores-1:
            yield df_.iloc[batch_size*i:cumulative+rest]
        else:
           yield df_.iloc[batch_size*i:cumulative]


def worker_func(_df, model, le_itm):
    for row in _df.itertuples():
        predictions = model.predict(row.seq)
        item_ids= (-predictions).argsort()[:K] 
        item_ids = le_itm.inverse_transform(item_ids)-1
        # indices = np.argpartition(preds, -K)[-K:]
        # best_movie_ids = indices[np.argsort(preds[indices])]
        # item_ids = le_itm.inverse_transform(best_movie_ids) +1
        sdict[row.utrip_id] = list(item_ids) 
    
    
def write_mongo_parallel(_df,model,le_itm):
    processes = [ multiprocessing.Process(target=worker_func, 
                                      args=(d,model, le_itm )) for d in split_df(_df)]

    for process in processes:
        process.start()
    
    for process in processes:
        process.join()


def evaluate_accuracy_at_4(submission,ground_truth):
    '''checks if the true city is within the four recommended cities'''
    data_to_eval = submission.join(ground_truth,on='utrip_id')
    hits = data_to_eval.apply(
        lambda row: row['city_id'] in (row[['city_id_1', 'city_id_2', 'city_id_3', 'city_id_4']].values),
            axis = 1)
    return hits.mean()    


def evaluate_lstm_model(hyperparameters, train, test,random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='lstm',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    #val_mrr = sequence_mrr_score(model, validation)
    # test_rec_prec = sequence_precision_recall_score(model, test, k=4, exclude_preceding=True )[0]
    # val_rec_prec = sequence_precision_recall_score(model, validation, k=4, exclude_preceding=True )[0]

    return test_mrr, model  

def evaluate_pooling_model(hyperparameters, train, test, random_state):

    h = hyperparameters

    model = ImplicitSequenceModel(loss=h['loss'],
                                  representation='pooling',
                                  batch_size=h['batch_size'],
                                  learning_rate=h['learning_rate'],
                                  l2=h['l2'],
                                  n_iter=h['n_iter'],
                                  use_cuda=CUDA,
                                  random_state=random_state)

    model.fit(train, verbose=True)

    test_mrr = sequence_mrr_score(model, test)
    #val_mrr = sequence_mrr_score(model, validation)
    #test_rec_prec = sequence_precision_recall_score(model, test)[0]
    #val_rec_prec = sequence_precision_recall_score(model, validation)[0]
    
    return test_mrr, model


max_sequence_length = 15
min_sequence_length = 3
step_size = None
random_state = np.random.RandomState(100)

#dataset = get_movielens_dataset('1M')

DEBUG = True

data_path = 'https://www.dropbox.com/s/0praxm2h7k6xv6i/booking_train_set.csv?dl=1'
#data_path = 'https://www.dropbox.com/s/hjblv5by6xtte50/booking_test_set.csv?dl=1'

cols = ['utrip_id','affiliate_id','checkin','hotel_country','city_id']

nrows = 40_000 if DEBUG else None
df = pd.read_csv(data_path,parse_dates=["checkin"],infer_datetime_format=True,usecols=cols,nrows=nrows)



df["row_num"] = df.groupby("utrip_id")["checkin"].rank(ascending=True,pct=False).astype(int)
utrip_counts = df["utrip_id"].value_counts()
df["total_rows"] = df["utrip_id"].map(utrip_counts)
df["last"] = (df["row_num"] ==df["total_rows"])#.astype(int)
df = df.loc[df["total_rows"]>=3]
###
df = df.loc[(df[["city_id","utrip_id","last"]].shift() != df[["city_id","utrip_id","last"]]).max(axis=1)]
df = df.reset_index(drop=True)
######

# ### replace rare variables (under 2 occurrences) with "-1" dummy OR with their hotel country! (as negative mapped number)
city_ids_counts = df["city_id"].value_counts()
print("before uniques",df["city_id"].nunique())
city_ids_counts = city_ids_counts.to_dict()
df["city_id"] = df["city_id"].where(df["city_id"].map(city_ids_counts)>2, -1)
#### ###
le_usr = preprocessing.LabelEncoder() # user encoder
le_itm = preprocessing.LabelEncoder() # item encoder
df['item_id'] = (le_itm.fit_transform(df['city_id'])+1).astype('int32') ### +1 ?
df['user_id'] = (le_usr.fit_transform(df['utrip_id'])).astype('int32')


# train_inds, test_inds = next(GroupShuffleSplit(test_size=.35, n_splits=2,random_state = 0).split(df,groups=df['utrip_id']))

# X_test = df.iloc[test_inds]
# test_no_last = X_test.loc[X_test["last"]==False].drop(["last"],axis=1)
# train = df.iloc[train_inds]

# last_city = pd.concat([X_test.loc[X_test["last"]==True]]).drop(["last"],axis=1)


train_inds, test_inds = next(GroupShuffleSplit(test_size=.3, n_splits=2,random_state = 0).split(df,groups=df['utrip_id']))

X_test = df.iloc[test_inds]

# train = pd.concat([df.iloc[train_inds],
#                   X_test.loc[X_test["last"]==False]]).drop(["last"],axis=1) # ORIG
train = df.copy() # new dan = "train + test"

last_city = pd.concat([X_test.loc[X_test["last"]==True]]).drop(["last"],axis=1)





train = Interactions(user_ids=train['user_id'].values,
                           item_ids=train['item_id'].values,
                           #ratings=df['rating'].values,
                           timestamps=train['checkin'].values)


train, test = user_based_train_test_split(train,
                                              random_state=random_state)

train = train.to_sequence(max_sequence_length=max_sequence_length,
                              min_sequence_length=min_sequence_length,
                              step_size=step_size)

test = test.to_sequence(max_sequence_length=max_sequence_length,
                            min_sequence_length=min_sequence_length,
                            step_size=step_size)

hp = {'n_iter': 15, 'loss': 'hinge', 'learning_rate': 0.01, 'l2': 0.0, 'embedding_dim': 64, 'batch_size': 16}
# Best lstm result: {'test_mrr': 0.2304, 'validation_mrr': 0.256,
hp_lstm = {'n_iter': 15, 'loss': 'adaptive_hinge', 'learning_rate': 0.05, 'l2': 0.0001, 'embedding_dim': 64, 'batch_size': 512}
test_mrr, model = evaluate_lstm_model(hp_lstm, train, test,random_state)
test_mrr, model = evaluate_pooling_model(hp, train, test, random_state)
print(test_mrr.mean())




K = 4
cols = ['utrip_id','city_id','user_id','item_id','checkin']

test_no_last = X_test.loc[X_test["last"]==False]

#df_pred = test_no_last.groupby('utrip_id',as_index=False).agg({'item_id':'unique'}).rename(columns={'item_id':'seq'})
df_pred = test_no_last.groupby('utrip_id').apply(lambda x: list(x['item_id'])).rename('seq').reset_index()

manager = multiprocessing.Manager()
sdict = manager.dict()
write_mongo_parallel(df_pred, model, le_itm)
sdict = dict(sdict)


citys_cols = [f'city_id_{i}' for i in range(1,K+1)]
recom_df = pd.DataFrame(sdict).T
recom_df.columns = citys_cols
recom_df = recom_df.reset_index().rename(columns={'index':'utrip_id'})

#recom_df = recom_df.merge(last_city[['utrip_id','city_id']],how='left',on='utrip_id')
ground_truth = last_city[['user_id','utrip_id','city_id']].set_index('utrip_id')



#evaluate_accuracy_at_4(recom_df,ground_truth)
res =  recom_df.merge(last_city[['utrip_id','city_id']],how='left',on='utrip_id')
res['city_id'] = res['city_id'] 
res['pred'] = res.apply(lambda row: row['city_id'] in (row[citys_cols].values),axis = 1)
np.mean(res['pred'])#top@8 0.10781010719754977
a=res[res.pred==True]
