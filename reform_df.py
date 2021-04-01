from datetime import timedelta
import numpy as np


def reform_df (r_df):  # combine concurrent rows that share (non-date) values

    first_row = False
    collect=[]

    for i, row in r_df.iterrows():
        if first_row == False:
            first_row = True
            collect.append(row)
        else:
            if row['city_id'] == collect[-1]['city_id'] and \
            row['affiliate_id'] == collect[-1]['affiliate_id'] and \
            row['device_class'] == collect[-1]['device_class'] :
                collect[-1]['checkout'] = row['checkout']
            else:
                collect.append(row) 
    re_df = pd.DataFrame(collect) 
    return re_df
                