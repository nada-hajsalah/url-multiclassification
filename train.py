from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle


def run_train(x, y):
    """
    Function to train train model: xgboost
    :param x:
    :param y:
    :return:
    """
    print('***Training***')
    clf = OneVsRestClassifier(XGBClassifier(importance_type='gain', n_estimators=100))
    clf.fit(x, y)
    with open('classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
        print("***Training is done****")
