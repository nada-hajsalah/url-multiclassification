from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn import metrics
import pickle


def test(X_test, vect, model_path='classifier.pkl'):
    """

    :param X_test:
    :param vectorizer:
    :param model_path:
    :return:
    """
    with open(model_path, 'rb') as f:
        classifier = pickle.load(f)
    print("Test")

    Y_pred = classifier.predict(X_test)
    y_pred = vect.inverse_transform(Y_pred)

    return y_pred


def metrics(y_test_multilabel, predictions):
    """

    :param y_test_multilabel:
    :param predictions:
    :return:
    """
    print("Accuracy :", metrics.accuracy_score(y_test_multilabel, predictions))
    print("Hamming loss ", metrics.hamming_loss(y_test_multilabel, predictions))
    precision = precision_score(y_test_multilabel, predictions, average='micro')
    recall = recall_score(y_test_multilabel, predictions, average='micro')
    f1 = f1_score(y_test_multilabel, predictions, average='micro')

    print("\nMicro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    precision = precision_score(y_test_multilabel, predictions, average='macro')
    recall = recall_score(y_test_multilabel, predictions, average='macro')
    f1 = f1_score(y_test_multilabel, predictions, average='macro')

    print("\nMacro-average quality numbers")
    print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
    print("\nClassification Report")
    print(metrics.classification_report(y_test_multilabel, predictions))
