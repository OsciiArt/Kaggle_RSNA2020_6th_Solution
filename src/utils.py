import os, glob, pickle, time, gc, copy


def pickle_save(path, df):
    with open(path, 'wb') as f:
        pickle.dump(df, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        df = pickle.load(f)
    return df

def ri(df):
    return df.reset_index(drop=True)