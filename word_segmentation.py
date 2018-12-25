#-*- encoding:utf-8 -*-  \
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
import jieba
jieba.enable_parallel(20)

ORI_ROOT = "/home/dxx/Backup/STUDY/DM//data/train/"
SEG_ROOT = "/home/dxx/Backup/STUDY/DM/data/train_cut/"

def get_stop_words():
    with open('stop_words_ch.txt') as f:
        stopwords = [ x.strip() for x in f.readlines() ]
    # [ lambda x: print(x) for x in stopwords]
    return stopwords
stopwords = get_stop_words()


def data_process(in_dir, out_dir):
    'word segmentation for all txt file in `root` directory'
    for f in os.listdir(in_dir):
        in_file = os.path.join(in_dir, f)
        out_file = os.path.join(out_dir, f)
        split_sentence(in_file, out_file )


def split_sentence(in_file, out_file):
    with open(in_file, 'r') as fin, open(out_file, 'w') as fout:
        for line in fin.readlines():
            words = jieba.cut(line.strip())
            words = [ x for x in words if x.strip() and x not in stopwords ]
            fout.write( ' '.join(words) + '\n' )


def main():
    for category in os.listdir(ORI_ROOT):
        in_dir = os.path.join(ORI_ROOT, category)
        out_dir = os.path.join(SEG_ROOT, category)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        data_process(in_dir, out_dir)

if __name__ == '__main__':
    main()
    # data_process('tmp1', 'tmp2')
