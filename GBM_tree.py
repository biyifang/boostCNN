import xgboost as xgb
import numpy as np 
import random

seed = 1111

np.random.seed(seed)
random.seed(seed)

train_embedding = np.load('train_embedding.npy')
train_label = np.load('train_label.npy') 
val_embedding = np.load('val_embedding.npy')
val_label = np.load('val_label.npy') 

dtrain = xgb.DMatrix(train_embedding, label = train_label)
dval = xgb.DMatrix(val_embedding, label = val_label)

param = {'max_depth': 3, 'eta': 0.7, 'objective': 'multi:softprob', 'num_class': 10}

watchlist = [(dval, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dval)
labels = dval.get_label()
print('error=%f' %(np.sum(np.argmax(preds,1) == labels)/float(len(labels))))


