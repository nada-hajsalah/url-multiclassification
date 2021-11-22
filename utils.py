import os
import glob
import nltk
import re
import pandas as pd
import tldextract
import logging
import torch
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from urllib.parse import urlparse, unquote
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

logging.basicConfig(level=logging.WARNING)
nltk.download('stopwords')


def load_dataset(path):
    """
    Read all files in the dataset and convert them into a dataframe
    :param path: path of the fold containing dataset
    :return: dataframe
    """

    print("\n--Loading data..")
    all_files = glob.glob(os.path.join(path, "*.parquet"))
    if not all_files:
        raise ValueError('The folder is empty. Please enter another path')
    df_from_each_file = (pd.read_parquet(f, engine='pyarrow') for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    print("Data loaded successfully..")
    print("Number of rows in the original dataset:", len(concatenated_df))
    return concatenated_df


def remove_duplicates(df, mode):
    """
    Check if there are duplicated rows with respect to the 'url' column in the dataset. Two possibilities :
        1/ the url and the target are duplicated
        2/ the url is duplicated with different targets

    :param df: dataframe with three columns 'url', 'target', 'day'
    :param mode: string, choices=('last', 'intersection'), indicates the mode to use to remove the duplicates.
            'last' : keep the last entry (in the dataset) from the duplicates of each given 'url'
            'intersection' : consider the intersection of the different 'target's for each given duplicated 'url'.
    :return: df_no_dup: dataframe without duplicates
    """

    exists_duplicate = df.duplicated(subset=['url']).any()
    print("\n--Removing potential duplicates from the given dataset..")

    if exists_duplicate:
        if mode == "last":
            df_no_dup = df.drop_duplicates(subset=["url"], keep="last", inplace=False)

        elif mode == "intersection":
            # to take into account the intersection between the different targets of same url
            df["target"] = df["target"].apply(set)
            df_no_dup = pd.DataFrame(df.groupby('url', as_index=True)["target"].agg(lambda x: set.intersection(*x)))
            df_no_dup.reset_index(level=0, inplace=True)
            df_no_dup["target"] = df_no_dup["target"].apply(list)

        else:
            raise ValueError("Failed to remove duplicated rows. %s mode is not recognized. Please choose one of these "
                             "modes ('last', 'intersection')" % mode)

        print("Total number of duplicate entries removed from the given dataset: ", len(df) - len(df_no_dup))

    else:
        print("There are no duplicate entries in the given dataset.")
        df_no_dup = df

    print("Number of rows considered in the dataset:", len(df_no_dup))
    print("Removing duplicates successfully done..")
    return df_no_dup


def remove_stopwords_and_tokens(sentence):
    """
    Remove stopwords from sentence and length less than 2
    :param sentence:
    :return:
    """

    stopwords_list = stopwords.words('french')
    tokens = sentence.split(" ")
    tags = ["html", "php", "htm", "phtml"]
    tokens_filtered = [word for word in tokens if
                       (not word in stopwords_list) and (len(word) > 2) and (not word in tags)]
    return " ".join(tokens_filtered)


def clean_path(path):
    """
    Clean path: remove stopwords, punctuations, digits, token with length<2, space
    :param path:
    :return: path: clened path
    """

    # remove punctuation and digits
    path = re.sub(r'[^\w\s]|[0-9]', ' ', path)

    # remove stopwords:
    path = remove_stopwords_and_tokens(path)

    # remove extra space
    path = " ".join(path.split())
    return path


def parse_url_clean(df):
    """
      Function for parsing  and cleaning the url
      :param df: Dataframe
      :return: Dataframe with new columns as features
    """

    # Step 1 : Parse, decode and clean path
    df["Path"] = df["url"].apply(lambda x: urlparse(x).path)
    df["Path"] = df["Path"].apply(lambda x: unquote(x))
    df["Path"] = df["Path"].apply(lambda x: clean_path(x))

    # Step 2 : Parse TLD (top-level domain)
    df['tld'] = df['url'].apply(lambda x: tldextract.extract(x).suffix)  # We consider only the last part of the tld
    # 1/ parse and preprocessing of netloc
    # extract name of domain name
    df["Netloc"] = df['url'].apply(lambda x: urlparse(x).netloc)
    # remove wwww and tld from netloc
    url = re.compile(r"(www)?[\.]")
    df["Netloc"] = df["Netloc"].apply(lambda x: url.sub(' ', x).strip())
    df["Netloc"] = df.apply(lambda x: x.Netloc.replace(x.tld, ""), axis=1)

    return df


def load_tld_files(path):
    """
    Load files contained tld and their types
    :param path:
    :return: dict with tld as keys and type as value
    """

    tld = pd.read_csv(path + "/" + "tld-list-details.csv")
    gtld = pd.read_excel(path + "/" + "gtld.xlsx")
    gtld_dict = dict(zip(gtld.TLD, gtld.Catégorie))
    tld_dict = dict(zip(tld.TLD, tld.Type))

    return gtld_dict, tld_dict


def tld_category(tld, gtld_dict, tld_dict):
    """
    Return the category of the top level domain
    :param tld: top level domain
    :param gtld_dict:
    :param tld_dict:
    :return:
    """

    if tld.upper() in gtld_dict:
        return gtld_dict[tld.upper()]
    elif tld in tld_dict:
        return tld_dict[tld]
    else:
        return "other_tld"


def extract_features(df, gtld_dict, tld_dict):
    """
     Function for feature engineering: extract features from url

      :param gtld_dict:
      :param tld_dict:
      :param df: Dataframe
      :return: Dataframe with new columns as features
    """
    df_features = df.copy()
    # extract length of url
    length_url_median = df_features['url'].apply(len).median()
    df_features['is_long'] = df_features['url'].apply(lambda x: len(x) > length_url_median).astype(int)

    # extract tld
    df_features["tld_category"] = df_features["tld"].apply(lambda x: tld_category(x, gtld_dict, tld_dict))

    return df_features


def collate_batch(batch):
    """

    :param batch:
    :return:
    """
    word_tensor = torch.tensor([[1.], [0.], [45.]])
    label_tensor = torch.tensor([[1.]])

    text_list, classes = [], []
    for (_text, _class) in batch:
        text_list.append(word_tensor)
        classes.append(label_tensor)
    text = torch.cat(text_list)
    classes = torch.tensor(classes)
    return text, classes


def onehotcoding(x, mode):
    """

    :param x: vector
    :param mode: test or train
    :return:
    """
    if mode == "test":
        with open('onehotencoder.pkl', 'rb') as f:
            onehot_encoder = pickle.load(f)
    if mode == "train":
        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(x)
        with open("onehotencoder.pkl", "wb") as f:
            pickle.dump(onehot_encoder, f)
    X = onehot_encoder.transform(x)
    return X


def get_vector(x, mode):
    """

    :param x:
    :param mode:
    :return:
    """
    if mode == "train":
        vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(
            1, 3), norm='l2', max_df=0.8, max_features=10000)
        vectorizer.fit(x)
        with open('vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
    else:
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)

    url_array = vectorizer.transform(x)

    return url_array


def get_labels(targets):
    """
    Return binary labels
    :param targets:
    :return:vectorizer
    :return: binary targets
    """

    def tokenizer(x):
        return x
    #max features =5 because EDA show that the average number of features is 5
    vectorizer = CountVectorizer(max_features=5, lowercase=False, tokenizer=tokenizer, binary=True).fit(
        targets)
    y_multilabel = vectorizer.transform(targets)

    print('binary targets uploaded ')

    return vectorizer, y_multilabel


def input_classifier(df, mode):
    """
    Return dataframe as input for the classifier
    :param df:
    :param mode:
    :return:
    """
    df["text_vector"] = df["Netloc"] + " " + df["Path"]
    u = get_vector(df["text_vector"], mode)
    # categorical tld

    enc_df = pd.DataFrame(u.toarray())
    # merge with main df  on key values
    df_final = df.join(enc_df)

    # drop other columns

    return df_final, u
