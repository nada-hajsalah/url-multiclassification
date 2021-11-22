from torch.utils.data import Dataset
from utils import *


class TextDataset(Dataset):
    def __init__(self, path, path2, mode='last'):
        # load dataset and tld files
        df = load_dataset(path)
        gtld_dict, tld_dict = load_tld_files(path2)

        # remove duplicates
        df = remove_duplicates(df, mode)

        # parse url
        df = parse_url_clean(df)
        # extract features
        df = extract_features(df, gtld_dict, tld_dict)
        print(df.head(3))

        self.data = df
        self.text = self.data.Netloc + " " + self.data.Path
        self.labels = self.data.target

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Class": label}
        return sample
