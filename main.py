import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import DataConvertionWarning
warnings.filterwarnings(action='ignore', category=DataConvertionWarning)

pd.set_option('display.float_format', lambda x: '%.5f' % x)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

subject_1 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S006.csv')
subject_2 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S008.csv')
subject_3 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S009.csv')
subject_4 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S010.csv')
subject_5 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S012.csv')
subject_6 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S013.csv')
subject_7 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S014.csv')
subject_8 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S015.csv')
subject_9 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S016.csv')
subject_10 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S017.csv')
subject_11 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S018.csv')
subject_12 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S019.csv')
subject_13 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S020.csv')
subject_14 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S021.csv')
subject_15 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S022.csv')
subject_16 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S023.csv')
subject_17 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S024.csv')
subject_18 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S025.csv')
subject_19 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S026.csv')
subject_20 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S027.csv')
subject_21 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S028.csv')
subject_22 = pd.read_csv(r'C:\Users\Abhiram P\Desktop\Work\harth\S029.csv')

subject_final = pd.concat([subject_1,subject_2,subject_3,subject_4,subject_5,subject_6,subject_7,subject_8,
                     subject_9,subject_10,subject_11,subject_12,subject_13,subject_14,subject_15,subject_16,
                     subject_17,subject_18,subject_19,subject_20,subject_21,subject_22])

df_final = pd.DataFrame(subject_final)

df = df_final.copy()
df = df.drop_duplicates()
df = df.drop(['index', 'Unnamed: 0'], axis=1)
df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9
x_final = df.drop('label', axis=1)
y_final = df[['label']]

x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(x_final, y_final,
                                                                            test_size=0.2, random_state=42)

KNC = KNeighborsClassifier()
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}
knc_gridsearch = GridSearchCV(KNC, param_grid, cv=5, scoring='accuracy', n_jobs=4, verbose=100)
knc_gridsearch.fit(x_train_final, y_train_final)
best_knc = knc_gridsearch.best_estimator_
KNC_ypred = best_knc.predict(x_test_final)
print(classification_report(KNC_ypred, y_test_final))

KNC_score = best_knc.score(x_test_final, y_test_final)
import joblib

joblib.dump(best_knc, 'KNC.pkl')
loaded_knc = joblib.load('KNC.pkl')

KNC_pred = loaded_knc.predict(x_test_final)
score = loaded_knc.score(x_test_final, y_test_final) * 100
