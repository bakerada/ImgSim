from simtool.serving import ModelServer
from utils.helpers import read_config
import pytest
import numpy as np

@pytest.fixture()
def config():
	return 'config/deploy.json'

class TestModelServer:
	def test_initialize_server(self, config):
		server = ModelServer(config)

	def test_query_topk_img(self,config):
		server = ModelServer(config)
		path = 'test/data/KO5OR.jpg'
		nbrs = server.query_topk(path, 10)
		assert len(nbrs) == 1
		assert len(nbrs[0]) ==10

	def test_query_topk_dir(self,config):
		server = ModelServer(config)
		path = 'test/data/test_dir'
		nbrs = server.query_topk(path, 10)
