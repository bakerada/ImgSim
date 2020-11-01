from pathlib import Path
import argparse
import os
import random
import numpy as np 
from shutil import copy

def get_classes(path):
	return os.listdir(str(path))

def get_split_counts(path,test_ratio,eval_ratio):
	# Find the number of files to sample from each class to create the test and eval sets
	classes = get_classes(path)
	total_imgs = len(list(path.glob('**/*')))
	samples_per_class = total_imgs // len(classes)
	test_count = samples_per_class * test_ratio
	eval_count = samples_per_class * eval_ratio
	return int(test_count), int(eval_count)

def index_list(lst,obj):
	return [obj[i] for i in lst]

def sample_dir(path,test_count,eval_count):
	files = list(path.glob('*'))
	splits = [len(files) - (test_count + eval_count),test_count,eval_count]
	index = list(range(len(files)))
	random.shuffle(index)
	samples = np.split(index, np.cumsum(splits))

	return {'train': index_list(samples[0],files),
	        'test': index_list(samples[1],files),
	        'eval': index_list(samples[2],files)}

def copy_files(samples, destination, cls):
	for k,v in samples.items():
		folder = destination / k / cls
		folder.mkdir(parents=True, exist_ok=True)
		for i in v:
			copy(str(i),str(folder))

def split_data(rootdir, savedir, test_ratio=0.2, eval_ratio=0.05):
	test_count,eval_count = get_split_counts(rootdir, test_ratio, eval_ratio)
	classes = get_classes(rootdir)
	for c in classes:
		samples = sample_dir(rootdir / c, test_count, eval_count)
		copy_files(samples, savedir, c)





if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Split data into train, test, and eval")
	parser.add_argument('--rootdir', type=str, help="Path to raw data")
	parser.add_argument('--savedir', type=str, help="Path to where splits are written")
	parser.add_argument('--test_ratio', type=float, default=0.2, help= "Percent of data split into test")
	parser.add_argument('--eval_ratio', type=float, default=0.2, help= "Percent of data split into eval")
	args = parser.parse_args()

	rootdir = Path(args.rootdir)
	savedir = Path(args.savedir)
	savedir.mkdir(parents=True, exist_ok=True)

	split_data(rootdir, savedir, args.test_ratio, args.eval_ratio)

	



