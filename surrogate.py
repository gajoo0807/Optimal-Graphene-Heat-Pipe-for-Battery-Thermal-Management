import numpy as np
import sys
import random
import scipy.stats as st
from math import e,cos
from scipy.integrate import quad
import sklearn.gaussian_process as gp
import seaborn as sns
import statsmodels.api as sm 
import pandas as pd
import os
from smt.sampling_methods import LHS,Random,FullFactorial
from smt.surrogate_models import RBF,KRG,KPLS,QP,LS,RMTB,IDW
import matplotlib.pyplot as plt
from smt.applications.mixed_integer import(
    FLOAT,
    ORD,
    ENUM,
    MixedIntegerSamplingMethod,
    MixedIntegerSurrogateModel,
    GOWER,
    MixedIntegerContext,
)
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pickle




class Surrogate_model():
    def __init__(self):
        # self.current_min=[]
        self.dim=15
        self.eval_times = 0
        self.reach=0
        self.xtypes=[FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT,FLOAT]
        self.xlimits=np.array([[0,10],[0,10],[0,10],[0,180],[0,360],[0,5],[0.05,10],[0.05,20],[0,3],[6,21],[30,70],[0.5,400],[0.5,400],[0,3],[20,40]])
        mixint = MixedIntegerContext(self.xtypes, self.xlimits)
        self.model = mixint.build_surrogate_model(LS(print_global=False))
    def latin_sampling(self, input_list):
        stir=input_list[0]
        shocking=input_list[1]
        pumpdelivery=input_list[2]
        immersion=input_list[3]
        heatDissipation=input_list[4]
        collwattage=input_list[5]
        diameter=input_list[6]
        height=input_list[7]
        axialHheating=input_list[8]
        longitudinalHeating=input_list[9]
        heatingWattage=input_list[10]
        ambientTemp=input_list[11]
        '''return the minimum of 100 points'''
        xlimits = np.array([[0,3],[0,4.5],[0,0.05],[stir,stir],[shocking,shocking],[pumpdelivery,pumpdelivery],[immersion,immersion],[heatDissipation,heatDissipation],[collwattage,collwattage],[diameter,diameter],[height,height],[axialHheating,axialHheating],[longitudinalHeating,longitudinalHeating],[heatingWattage,heatingWattage],[ambientTemp,ambientTemp]])
        sampling=MixedIntegerSamplingMethod(self.xtypes, xlimits, LHS, criterion="ese")
        x = sampling(200)
        x_pred=self.model.predict_values(x)
        current_min = x_pred[np.argmin(x_pred)]
        location=x[np.argmin(x_pred)]
        location=np.array(location).reshape(-1,15)
        return current_min,location
    def init_build(self, df_name):
        self.X_train=[]
        self.y_train=[]
        df=pd.read_csv(df_name)
        X=df.drop(['電池溫度','電阻(Ω)'],1)
        y=df['電池溫度']
        for i in range(len(df)):
            if y[i]!='--':                    # 電池沒短路
                solution=X.iloc[i]
                objective=float(y[i])-41
                self.X_train.append(np.array(solution))
                self.y_train.append(np.array(objective))
            else:
                for i in range(10):
                    X_new=self.add_data(X.iloc[i],df)                             # 電池短路
                    objective=30
                    self.X_train.append(np.array(solution))
                    self.y_train.append(np.array(objective))
        self.X_train=np.array(self.X_train)
        self.y_train=np.array(self.y_train)
        self.model.set_training_values(self.X_train,self.y_train)
        self.model.train()
    def add_data(self,dfin,df):
        df2=[]
        n = random.randint(1,15)
        randomlist = random.sample(range(0, 15), n)
        for i in range(len(dfin)):
            if dfin[i]==0:
                df2.append(0)
            elif i==9 or i==10:
                df2.append(dfin[i])
            elif i in randomlist:
                s=np.random.normal(df.iloc[:,i].mean(),0.03*df.iloc[:,i].std(),1)
                df2.append(s[0])
            else:
                df2.append(dfin[i])
        new_df=pd.DataFrame([df2], columns=['石墨烯濃度(wt%)','奈米氧化鋁濃度(wt%)','碳黑奈米濃度 (wt%)','電磁攪拌時間(min)','超音波震盪時間(min)','幫浦輸送速度 (L/min)','沉浸液總體積(L)','沉浸液於散熱區體積(L)','冷卻瓦數(W)','電池直徑(mm)','電池高度(mm)','加熱區軸向熱傳導係數(W/mK)','加熱區縱向熱傳導係數(W/mK)','加熱瓦數(W)','環溫(℃)'])
        return new_df
    def calculate(self, input_list):
        predict_min,new_sample=self.latin_sampling(input_list)
        new_sample=np.squeeze(new_sample)
        print(f'{new_sample=}')
        list_label=["石墨烯濃度(wt%)",'奈米氧化鋁濃度(wt%)','碳黑奈米濃度 (wt%)']
        print('the minimum we predict is')
        for index,element in enumerate(list_label):
            print(f'{element}: {new_sample[index]:.6}')
        print(f'the minimum is {(predict_min[0]+45):2.4f}')
        self.X_train=np.append(self.X_train,[new_sample],axis=0)
        return new_sample[0], new_sample[1], new_sample[2]
    
    def retrain(self, temp):
        y_sample = temp-45
        self.eval_times+=1
        self.y_train=np.append(self.y_train,y_sample)
        self.model.set_training_values(self.X_train,self.y_train)
    
    def run(self):
        
        self.eval_times=0
        self.init_build("給清華資工系_沉浸式冷卻_20220802.csv")
        while self.eval_times < 500:
            print('=====================FE=====================') 
            print(f'{self.eval_times=}')
            input_list = [0]*12
            input_list[0]=input("electromagnetic stirring time (min)")
            input_list[1]=input("超音波震盪時間 ")
            input_list[2]=input("幫浦輸送速度")
            input_list[3]=input("沉浸液總體積 ")
            input_list[4]=input("沉浸液於散熱區體積 ")
            input_list[5]=input("冷卻瓦數 ")
            input_list[6]=input("電池直徑")
            input_list[7]=input("電池高度: ")
            input_list[8]=input("加熱區軸向熱傳導係數: ")
            input_list[9]=input("加熱區縱向熱傳導係數")
            input_list[10]=input("加熱瓦數: ")
            input_list[11]=input("環境溫度: ")
            self.calculate(input_list)
            self.retrain(int(input("if battery is shorted,input 1:")))
            self.model.train()

    def save_model(self, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)     

if __name__ == '__main__':
    op=Surrogate_model()
    op.run()