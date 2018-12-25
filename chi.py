#-*- encoding:utf-8 -*-  
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
from collections import defaultdict

DATAPATH = '/home/dxx/Backup/STUDY/DM/data/train_cut/'
FEATUREPATH = '/home/dxx/Backup/STUDY/DM/data/train_chi/'

topk = 200
num_class = 6
tf_list = [ defaultdict(int) for _ in range(num_class) ]
chi_list = [ defaultdict(int) for _ in range(num_class) ]


def get_word_dic():
    'for each category, calculate TF(term-frequency)'
    for category in range(num_class):
        sub_dir = os.path.join(DATAPATH, str(category))
        for f in os.listdir(sub_dir):
            filename = os.path.join(sub_dir, f)
            with open(filename) as f:
                all_words = set()
                for line in f.readlines():
                    words = set(line.strip().split())
                    all_words = all_words | words
                for w in all_words:
                    tf_list[category][w] += 1
        tf_list[category]['num_docs'] = len( os.listdir(sub_dir) )


def calc_chi():
    'calculate chi value for (t,c)'
    # global chi_list
    for category in range(num_class):
        filename = os.path.join(FEATUREPATH, '{}_keywords.txt'.format(category))
        with open(filename, 'w') as f:
            for t in tf_list[category].keys():
                TC = tf_list[category][t]                   # A
                Tc = 0                                      # B
                for c in range(num_class):
                    if c == category:
                        continue
                    Tc += tf_list[c][t]
                tC = tf_list[category]['num_docs'] - TC     # C
                tc = 0                                      # D
                for c in range(num_class):
                    if c == category:
                        continue
                    tc += tf_list[c]['num_docs'] - tf_list[c][t]
                try:
                    val = (TC*tc-Tc*tC)**2  / ( (TC+Tc)*(tC+tc)*1.0 )
                except ZeroDivisionError:
                    val = 0.0
                chi_list[category][t] = val
            top_list = sorted(chi_list[category].items(), key=lambda t : t[1], reverse=True)
            keywords = [ pair[0] for pair in top_list[:topk] ]
            f.write(' '.join(keywords))


def main():
    if not os.path.exists(FEATUREPATH):
        os.makedirs(FEATUREPATH)
    get_word_dic()
    calc_chi()

if __name__ == '__main__':
    main()
