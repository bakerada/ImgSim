import pytest
from utils.helpers import read_config
from simtool.model import RockClassifier
from simtool.extract_embeddings import EmbeddingExtractor
import os

@pytest.fixture()
def extractor():
    config = read_config('config/deploy.json')
    extractor = EmbeddingExtractor(config['model_path'],
                                   batch_config = config['data'],
                                   **config['model_args'])
    return extractor

class TestEmbeddingExtractor:
    def test_load_model(self, extractor):
        assert not extractor.use_cuda

    def test_process_img(self,extractor):
        test_path = 'test/data/KO5OR.jpg'
        pred,emb,path = extractor._process_img(test_path)
        assert pred <=5
        assert len(emb.shape) == 1


    def test_process_dir(self,extractor):
        test_path = '/s/mlsc/abake116/geodata/eval'
        pred,emb,path= extractor._process_dir(test_path)
        assert pred.shape[0] == emb.shape[0]
        assert len(path) == pred.shape[0]

    def test_determine_method(self,extractor):
        test_path = '/s/mlsc/abake116/geodata/test/marble/KO5OR.jpg'
        _, is_file = extractor._determine_method(test_path)
        assert is_file
        test_path = '/s/mlsc/abake116/geodata/eval'
        _, is_file = extractor._determine_method(test_path)

    def test_save_img(self,extractor):
        test_path = 'test/data/KO5OR.jpg'
        save_path = 'test/data'
        extractor.extract_embeddings(test_path,save_dir=save_path)
        assert os.path.isfile('test/data/data_ko5or.pickle')

    def test_save_dir(self,extractor):
        test_path = 'test/data/test_dir'
        save_path = 'test/data/test_out'
        extractor.extract_embeddings(test_path,save_dir=save_path)
