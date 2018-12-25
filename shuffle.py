import random

ori_txt = '/home/dxx/Backup/STUDY/DM/data/train.txt'
shuffle_txt = '/home/dxx/Backup/STUDY/DM/data/train_shuffle.txt'

num_class = 6

sample_dict = { x:[] for x in range(num_class) }

target_sample = {
    0: 370,
    1: 8,
    2: 11,
    3: 15,
    4: 18,
    5: 16
}

with open(ori_txt) as f:
    for line in f.readlines():
        x = int(line[0])
        sample_dict[x].append(line)

shuffle_samples = []
for i in range(num_class):
    ori_sample = sample_dict[i]    
    rate = target_sample[i]//len(ori_sample) + 1
    ori_sample = ori_sample*rate
    shuffle_sample = ori_sample[:target_sample[i]]
    random.shuffle(shuffle_sample)
    shuffle_samples += shuffle_sample
random.shuffle(shuffle_samples)


with open(shuffle_txt, 'w') as f:
    f.writelines(shuffle_samples)
