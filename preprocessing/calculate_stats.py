import torch
from simtool.dataloader import *
from utils.helpers import read_config
import os


if __name__ == '__main__':
	config = {
		"loader": {
			"batch_size": 16,
			"shuffle": True,
			"num_workers": 2
		},
		"transforms": {
			"tensor": {},	
		}
		}
	data_loader = create_dataloader(os.environ['DATADIR'], config)

	mean = 0
	std = 0
	for img,_ in data_loader:
		view = img.view(img.size(0),img.size(1),-1)
		mean += view.mean(2).sum(0)
		std += view.std(2).sum(0)

	print("Mean Per Channel {}".format(mean / len(data_loader)))
	print("STD Per Channel {}".format(std / len(data_loader)))

