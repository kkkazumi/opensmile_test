import pandas as pd

emo_list=["SS_ternary","TS_ternary","TC_ternary","SS","TC1",
        "TC2","TC3","TC4","TC5","TS1","TS2","TS3","TS4","TS5"]

def drop_data(df,drop_list):
    return df.drop(drop_list,axis=1)

def get_acoustic_features(data):
    org_list=data.columns.to_list()
    list_data=pd.read_csv("data_list.csv")
    extract_list=list_data.columns.to_list()
    drop_list=set(org_list)^set(extract_list)
    _data=drop_data(data,drop_list)
    return _data

def get_emo_features(data):
    org_list=data.columns.to_list()
    drop_list=set(org_list)^set(emo_list)
    _data=drop_data(data,drop_list)
    return _data

if __name__ == "__main__":
    data=pd.read_csv("1911F2001.txt")
    print(get_acoustic_features(data))
    print(get_emo_features(data))
