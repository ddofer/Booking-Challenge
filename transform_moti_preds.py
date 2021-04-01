# coding: utf-8
import pandas as pd
df = pd.read_csv("train_predictions.csv",usecols=[1,2])

df["rank"] = df.groupby("user_id").transform("cumcount")

df_train = pd.read_csv("booking_train_set.csv",usecols=["utrip_id","city_id","checkin"],parse_dates=["checkin"],infer_datetime_format=True)
df_train.sort_values([ "checkin",
                "utrip_id"],inplace=True)

df_train["row_num"] = df_train.groupby("utrip_id")["checkin"].rank(ascending=True,pct=False).astype(int)
utrip_counts = df_train["utrip_id"].value_counts()
df_train["total_rows"] = df_train["utrip_id"].map(utrip_counts)

df_train = df_train.loc[df_train.row_num == df_train.total_rows]

df_train = df_train[["city_id","utrip_id"]]

df_train["label"] = 1

df.rename(columns={"Recom":"city_id","user_id":"utrip_id"},inplace=True)

df.merge(df_train,on=["utrip_id","city_id"],how="left").shape
df = df.merge(df_train,on=["utrip_id","city_id"],how="left")
df["label"].fillna(0,inplace=True)
df["rank"].isna().sum()

df.loc[df.label==1]["rank"].describe()

df.to_csv("list_booking_train.csv.gz",index=False,compression="gzip")

# get_ipython().run_line_magic('save', '"transform_moti_preds.py" 1-999999')
