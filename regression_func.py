import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge

from sklearn.preprocessing import OneHotEncoder


def regressType(types, path):
    df = pd.read_csv(path)

    df=df.drop(df[df['電池溫度']=='--'].index)
    df=df.reset_index()
    df.drop(["index"],inplace=True,axis=1)

    df = shuffle(df)
    train_data_size=int(len(df)*0.8)
    y_train=df['電池溫度'].values[:train_data_size]
    y_train=[float(item) for item in y_train]
    y_test=df['電池溫度'].values[train_data_size:]
    y_test=[float(item) for item in y_test]

    df=df.drop(['電阻(Ω)','電池溫度'],1)
    df=df.reset_index()
    df.drop(["index"],inplace=True,axis=1)
    train_data=df.values[:train_data_size, :]
    test_data=df.values[train_data_size:,:]
    X_train = pd.DataFrame(StandardScaler().fit_transform(train_data))
    X_test=pd.DataFrame(StandardScaler().fit_transform(test_data))
    cubic=PolynomialFeatures(degree=3)
    X_cubic_train=cubic.fit_transform(X_train)
    X_cubic_test=cubic.fit_transform(X_test)
    print('----info----')
    print(f'training data size:{len(X_train)}')
    print('testing data size:', len(X_test))
    score={}
    if types=='Lasso':
        print(f'---------Lasso ---------')
        models = Lasso(alpha=0.0)##,normalize=True) # 默認alpha =1 
    elif types=='Ridge':
        print(f'---------Ridge ---------')
        models=Ridge(alpha=0.0)#,normalize=True)
    models.fit(X_cubic_train,y_train)
    y_train_pred=models.predict(X_cubic_train)
    y_test_pred=models.predict(X_cubic_test)
    score['(MSE) train']=mean_squared_error(y_train,y_train_pred)
    score['(MSE) test ']=mean_squared_error(y_test,y_test_pred)
    score['(R^2) train']=r2_score(y_train,y_train_pred)
    score['(R^2) test ']=r2_score(y_test,y_test_pred)
    plt.scatter(
        y_test,
        abs(y_test-y_test_pred),
        c='red'
    )
    plt.ylabel('abs(predict-actual)')
    plt.xlabel('actual battery temperature')
    plt.title(f'{types}')
    plt.tight_layout()
    plt.savefig('compare.png')
    #plt.show()
    return score