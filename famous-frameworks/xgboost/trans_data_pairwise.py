import sys
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import xgbfir


def save_data(group_data,output_feature,output_group):
    if len(group_data) == 0:
        return

    output_group.write(str(len(group_data))+"\n")
    for data in group_data:
        # only include nonzero features
        feats = [ p for p in data[2:] if float(p.split(':')[1]) != 0.0 ]        
        output_feature.write(data[0] + " " + " ".join(feats) + "\n")


def trans_data(fi_name, output_feature_name, output_group_name):

    fi =  open(fi_name)
    output_feature = open(output_feature_name,"w")
    output_group = open(output_group_name,"w")
    
    group_data = []
    group = ""
    for line in fi:
        if not line:
            break
        if "#" in line: 
            line = line[:line.index("#")]
        splits = line.strip().split(" ")
        if splits[1] != group:
            save_data(group_data,output_feature,output_group)
            group_data = []
        group = splits[1]
        group_data.append(splits)

    save_data(group_data,output_feature,output_group)

    fi.close()
    output_feature.close()
    output_group.close()


if __name__ == "__main__":

    # Convert normal label-features file to label-features + group file
    trans_data("train.txt", "mq2008.train", "mq2008.train.group")

    trans_data("test.txt", "mq2008.test", "mq2008.test.group")

    trans_data("vali.txt", "mq2008.vali", "mq2008.vali.group")

    param = {
        "objective": "rank:pairwise", # learning objective
        "eval_metric": ["logloss", "map", "error", "ndcg"],
        "eta": 0.1, # learning_rate
        "gama": 1.0, # min_split_loss
        "min_child_weight": 0.1, # minimum sum of instance weight(hessian) needed in a child
        "max_depth": 6, # maximum depth of a tree
        "n_estimators": 44
    }

    # load train/valid/test data
    dtrain = xgb.DMatrix('mq2008.train')
    group_file = "mq2008.train.group"
    group = np.loadtxt(group_file)
    group = group.astype(int)
    dtrain.set_group(group)

    dvalid = xgb.DMatrix('mq2008.vali')
    group_file = "mq2008.vali.group"
    group = np.loadtxt(group_file)
    group = group.astype(int)
    dvalid.set_group(group)

    dtest = xgb.DMatrix('mq2008.test')
    group_file = "mq2008.test.group"
    group = np.loadtxt(group_file)
    group = group.astype(int)
    dtest.set_group(group)


    # Train train data and show metrics on valid data
    num_round = 88
    evals_result = {}
    bst_model = xgb.train(param, dtrain,
         evals=[(dvalid, "Valid")],
         num_boost_round=num_round,
         evals_result=evals_result)
    print(evals_result)

    # show feature importance plot
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(bst_model, height=0.8, ax=ax)
    #plt.show()
    fig.savefig('feature_importance.png')

    # show feature importance table
    #fmap = bst_model.get_score(importance_type='cover')
    #print(fmap)
    fmap = bst_model.get_score(importance_type='gain')
    print(fmap)
    #fmap = bst_model.get_score(importance_type='weight')
    #print(fmap)


    # saving to file with proper feature names
    xgbfir.saveXgbFI(bst_model, OutputXlsxFile='future_interaction.xlsx')

    # predict on test data
    preds = bst_model.predict(dtest)
    print(preds)



    bst_model.dump_model('model.txt') 



    






"""
../../xgboost mq2008.conf

../../xgboost mq2008.conf task=pred model_in=0004.model


# specify objective
objective="rank:pairwise"

# Tree Booster Parameters
# step size shrinkage
eta = 0.1 
# minimum loss reduction required to make a further partition
gamma = 1.0 
# minimum sum of instance weight(hessian) needed in a child
min_child_weight = 0.1
# maximum depth of a tree
max_depth = 6


# Task parameters
# the number of round to do boosting
num_round = 4
# 0 means do not save any model except the final round model
save_period = 0 
# The path of training data
data = "mq2008.train" 
# The path of validation data, used to monitor training process, here [test] sets name of the validation set
eval[test] = "mq2008.vali" 
# The path of test data 
test:data = "mq2008.test"      

"""
