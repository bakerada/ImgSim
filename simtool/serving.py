from simtool.indexing import Indexer
#from preprocessing.prep_data import *
from simtool.dataloader import *
from simtool.extract_embeddings import EmbeddingExtractor
from utils.helpers import read_config
import numpy as np
import os


class ModelServer:
    """ Method to serve the model, extracts embeddings and queries for NN.
        Compatible with Seldon Core
            Args:
                None: reads in config from env parameter CONFIG_PATH
    """

    def __init__(self):
        self.config = read_config(os.environ['CONFIG_PATH'])
        self.extractor = EmbeddingExtractor(self.config['model_path'],
                                   gpus=self.config['gpus'],
                                   batch_config = self.config['data'],
                                   **self.config['model_args'])

        self.indexer = Indexer()
        self.querier, self.imgpaths = self.prep_index()
    def prep_index(self):
        index, paths = self.indexer.fit(self.config['embedding_path'],
                             index_path=self.config.get('index_path',None))
        return index, np.array(paths)

    def _format_query(self, query):
        outputs = []
        for i in range(len(query)):
            imgpaths = self.imgpaths[query[i][0]]
            outputs.append(list(zip(imgpaths,query[i][1].astype(float))))
        return outputs

    def query_topk(self, path, k, return_embeddings=False):
        _, embedding, path = self.extractor.extract_embeddings(path)
        if len(embedding.shape) == 1:
            embedding = np.expand_dims(embedding,0)
        nbs = self.querier.knnQueryBatch(embedding, k=k, num_threads=self.indexer.num_threads)
        return self._format_query(nbs)
