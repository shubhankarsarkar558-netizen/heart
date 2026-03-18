import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
df=pd.read_csv('heart_disease.csv')
df.pop('Patient_ID')
# print(df)
# print(df.isnull().sum())

le=LabelEncoder()
df['Gender']=le.fit_transform(df['Gender'])
df['Smoking']=le.fit_transform(df['Smoking'])
df['Diabetes']=le.fit_transform(df['Diabetes'])
# print(df)
y=df[['Heart_Disease']]
# print(y)
x=df.drop(['Heart_Disease'],axis=1)
# print(x)
model=LinearRegression()
model.fit(x,y)
m1,m2,m3,m4,m5,m6,m7,m8=model.coef_[0]
c=model.intercept_
print(m1,m2,m3,m4,m5,m6,m7,m8,c)

Age=int(input("Enter Age:"))
Gender=int(input("Enter Gender:"))
Blood_Pressure=int(input("Enter Blood_Pressure:"))
Cholesterol=int(input("Enter Cholesterol:"))
Heart_Rate=int(input("Enter Heart_Rate:"))
Smoking=int(input("Enter Smoking:"))
Diabetes=int(input("Enter Diabetes:"))
BMI=float(input("Enter  BMI:"))
prediction_heart_disease=m1*Age+m2*Gender+m3*Blood_Pressure+m4*Cholesterol+m5*Heart_Rate+m6*Smoking+m7*Diabetes+m8*BMI+c
print(prediction_heart_disease)