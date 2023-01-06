import opensmile
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import joblib #to save model
import os
import glob

#with open('IS09_emotion.conf', 'w') as fp:
#    fp.write(config_str)

smile = opensmile.Smile(
    feature_set='/home/kazumi/download/opensmile-3.0-linux-x64/config/is09-13/IS09_emotion.conf',
    #feature_set=opensmile.FeatureSet.GeMAPSv01b,
    #feature_set=opensmile.FeatureSet.GeMAPSv01b,
    #feature_level=opensmile.FeatureLevel.Functionals,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors_Deltas,
)

def get_feature(wav_file_path):
    feat = smile.process_file(wav_file_path)
    return feat.values

model_filename = "./sample_model.joblib"

if(os.path.isfile(model_filename)):
    print("yes")
    clf=joblib.load(model_filename)
else:
    print("no")
    senti_df = pd.read_csv('./audio_speech_sentiment/TRAIN.csv')

    dic = {"Negative":0, "Neutral": 1, "Positive": 2}
    X, y = [], []
    for Filename, Class in senti_df[['Filename', 'Class']].values:
        feat = smile.process_file(f'./audio_speech_sentiment/TRAIN/{Filename}')
        X.append(feat.values)
        y.append(dic[Class])

    X = np.vstack(X)
    y = np.array(y)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    clf = RandomForestClassifier(max_depth=4, random_state=0)
    clf.fit(X, y)

    joblib.dump(clf,model_filename)

dir_name='./21sep_recording/B9001/audio/ML/*.wav'
files = glob.glob(dir_name)
for filepath in files:
    X_wav = []
    X_wav.append(get_feature(filepath)[0])
    y_test_pred = clf.predict_proba(X_wav)
    #y_test_pred = clf.predict(X_wav)
    print(filepath,"estimated result:", y_test_pred)  # 0.8
