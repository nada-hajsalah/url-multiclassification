import argparse

import scipy
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split

from dataset import TextDataset
from train import *
from utils import *
from eval import *

# logging.basicConfig(level=logging.WARNING)


def main(args):
    urls_dataset = TextDataset(args.path1, args.path2, "last")
    train_set, test_set = train_test_split(
        urls_dataset.data[:100], test_size=0.2, random_state=42)
    if args.mode == "train":
        # training
        v, y_train = get_labels(train_set["target"])

        print(train_set.head())
        df_final, url_vector = input_classifier(train_set, args.mode)
        df_final = df_final.drop(
            ["target", "url", "day", "Path", 'Netloc', 'Path', 'tld_category', 'tld', "text_vector"], axis=1)
        print(df_final.head())
        df_final = scipy.sparse.csr_matrix(
            df_final.astype(float).values)
        X_train = hstack((url_vector, df_final))
        print("ok")
        run_train(X_train, y_train)

        x_test, url_vector = input_classifier(train_set, "test")
        x_final = x_test.drop(
            ["target", "url", "day", "Path", 'Netloc', 'Path', 'tld_category', 'tld', "text_vector"], axis=1)
        x_final = scipy.sparse.csr_matrix(
            x_final.astype(float).values)
        X_test = hstack((url_vector, x_final))
        y_pred=test(X_test, v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train")

    parser.add_argument("--path1", type=str, default='data', help="path to the directory containing the dataset")
    parser.add_argument("--path2", type=str, default='tld',
                        help="path to the directory containing the tld files")
    args = parser.parse_args()
    main(args)
