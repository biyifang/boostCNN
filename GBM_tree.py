import xgboost as xgb
import numpy as np 
import random

seed = 1111

np.random.seed(seed)
random.seed(seed)

train_embedding = np.load('train_embedding')
train_label = np.load('train_label')
val_embedding = np.load('val_embedding')
val_label = np.load('val_label')

dtrain = xgb.DMatrix(train_embedding, label = train_label)
dval = xgb.DMatrix(val_embedding, label = val_label)

param = {'max_depth': 3, 'eta': 1, 'objective': 'multi:softprob'}

watchlist = [(dval, 'eval'), (dtrain, 'train')]
num_round = 10
bst = xgb.train(param, dtrain, num_round, watchlist)
preds = bst.predict(dval)
labels = dval.get_label()
print('error=%f' %
      (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i]) /
       float(len(preds))))
