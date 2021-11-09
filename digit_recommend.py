#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'mnist_train.csv')
#%%
data=df.values
#splitting of labels and values
x=data[:,1:]
y=data[:,0]
print(x.shape,y.shape)

#%%
def distance(x1,x2):
    return np.sqrt((sum(x1-x2)**2))
def KNN(x,y,querypoint,k=5):
    vals=[]
    m=x.shape[0]
    for i in range(m):
        d=distance(querypoint,x[i])
        vals.append((d,y[i]))
    vals.sort()
    vals=vals[:k]
    vals=np.array(vals)
    new_vals=np.unique(vals[:,1],return_counts=True)
    index=new_vals[1].argmax()
    pred=new_vals[0][index]
    return pred

