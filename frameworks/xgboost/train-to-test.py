import xgboost as xgb

# read in data
dtrain = xgb.DMatrix('test-train.txt')

dtest = xgb.DMatrix('test-test.txt')

# specify parameters via map
param = {'max_depth':0,
	  'eta':1,
	  'silent':1,
	  'objective':'reg:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)


bst.dump_model('model.text')

bst.save_model('model.bin')

# make prediction
preds = bst.predict(dtest)
print(preds)

#print(xgb.plot_importance(bst))
#xgb.plot_tree(bst, num_trees=2)


'''
# load xgboost model
bst_new = xgb.Booster({'nthread':4}) #init model
bst_new.load_model("model.bin") # load data

# predict using loaded model
preds = bst_new.predict(dtest)
print(preds)
'''

