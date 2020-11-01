import pytest
from simtool.dataloader import *
from utils.helpers import read_config

@pytest.fixture()
def config():
	return read_config('config/train.json')

class TestDataLoader:
	def test_build_transforms(self,config):
		transformations = build_transforms(config['data']['transforms'])
		assert len(transformations) > 0

	def test_create_dataloader(self,config):
		dataloader = create_dataloader(config['traindir'],config['data'])

		assert len(dataloader) > 0
		img,label = next(iter(dataloader))
		assert img.size(1) == 3
		assert img.size(0) == int(config['data']['loader']['batch_size'])
