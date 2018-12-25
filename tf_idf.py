#-*- encoding:utf-8 -*-  
import sys, os
import numpy as np
from sklearn.feature_extraction.text import  TfidfVectorizer


DATAPATH = '/home/dxx/Backup/STUDY/DM/data/train_cut/'
FEATUREPATH = '/home/dxx/Backup/STUDY/DM/data/train_chi/'
MATRIXFILE = '/home/dxx/Backup/STUDY/DM/data/train.txt'

num_class = 6
num_feature = 80


def get_voc():
    voc = set()
    for category in range(num_class):
        filename = os.path.join(FEATUREPATH, '{}_keywords.txt'.format(category))
        with open(filename) as f:
            keywords = f.read().split()[:num_feature]
        voc = voc | set(keywords)
    return list(voc)
voc = get_voc()
vectorizer = TfidfVectorizer(vocabulary=voc)


def calc_tfidf():
    'calculate tf-idf features of each article'
    for category in range(num_class):
        corpus = []
        sub_dir = os.path.join(DATAPATH, str(category))
        for f in sorted(os.listdir(sub_dir)):
            filename = os.path.join(sub_dir, f)
            with open(filename) as f:
                content = f.read()
                corpus.append(content)
        X = vectorizer.fit_transform(corpus)
    
        with open(MATRIXFILE, 'a') as f:
            for sample in X.toarray():
                features = [str(category),]
                for index, value in enumerate(sample, 1):
                    features.append('{}:{}'.format(index, value))
                f.write(' '.join(features) + '\n')

def main():
    calc_tfidf()

if __name__ == '__main__':
    main()
