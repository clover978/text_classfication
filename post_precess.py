import os, shutil

DATAPATH = '/home/dxx/Backup/STUDY/DM/data/test/0/'
OUTPUTPATH = '/home/dxx/Backup/STUDY/DM/output/test_result/'
num_class = 6

filenames = sorted(os.listdir(DATAPATH))
with open('result.txt') as f:
    results = [ line.strip()[-1] for line in f.readlines() ]

for category in range(num_class):
    os.makedirs(os.path.join(OUTPUTPATH, str(category)))

for filename, rst in zip(filenames, results):
    src = os.path.join(DATAPATH, filename)
    dst = os.path.join(OUTPUTPATH, str(rst), filename)
    shutil.copy(src, dst)