import torch
import torch.utils.data as data
from pathlib import Path
import torchvision.transforms as T
import torchvision.datasets as datasets
from collections import OrderedDict
from PIL import Image

#Order Matters with transformations
transform_map = OrderedDict([
		("hflip", T.RandomHorizontalFlip),
		("vflip", T.RandomVerticalFlip),
		("tensor", T.ToTensor),
		("normalize", T.Normalize)
	])



def build_transforms(transforms):
	transformations = []
	for k,v in transform_map.items():
		if k in transforms.keys():
			transformations.append(
					v(**transforms[k])
				)
	return transformations

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def create_dataloader(path, config):
	transformations = build_transforms(config['transforms'])
	dataset = datasets.ImageFolder(
        path,
        T.Compose([
        	*transformations
        ]))

	return data.DataLoader(dataset,**config['loader'])



class DirectoryDataset(data.Dataset):
	''' This dataset will be used to load imgs from a folder '''
	def __init__(self, datadir, transforms={}):
		self.datadir = datadir
		self.files = list(Path(datadir).glob('*'))
		self.transforms = T.Compose(build_transforms(transforms))

	def __getitem__(self,idx):
		img = pil_loader(str(self.files[idx]))
		return self.transforms(img)
