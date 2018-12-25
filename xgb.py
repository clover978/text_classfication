import xgboost as xgb
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix

dtrain = xgb.DMatrix('/home/dxx/Backup/STUDY/syd/data/train.txt')
dtest = xgb.DMatrix('/home/dxx/Backup/STUDY/syd/data/test.txt')

param = {'silent':0, 
         'max_depth':10,
         'eta':0.1,
         'subsample':1.0,
         'min_child_weight':5,
         'col_sample_bytree':0.2,
         'objective':'multi:softmax',
         'num_class':6 }
num_round = 100
watchlist = [(dtrain, '@')]

def train():
    bst = xgb.train(param, dtrain, num_round, evals = watchlist)
    bst.save_model('xtrain.model')
    # bst.dump_model('model.txt')

def test():
    bst = xgb.Booster()
    bst.load_model('xtrain.model')
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    with open('result.txt', 'w') as f:
        for y, y_ in zip(labels, preds):
            f.write('label: {:d}\tprediction: {:d}\n'.format(y, y_))

    print(classification_report(labels, preds))
    print(confusion_matrix(labels, preds))

def main():
    train()
    test()

if __name__ == '__main__':
    main()