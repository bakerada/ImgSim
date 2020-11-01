import numpy as np
import os
from pathlib import Path
import pickle
import nmslib


class Indexer:
    """ A Method to fit or load a HNSW approximate neighest neighbor model
            Args:
                m (int): The maximum outgoing connections in the graph
                efc (int): Construction time/accuracy tradeoff
                num_threads (int): Threads to use in fitting and querying
                metric (str): Similarity measurement ie Cosine or L2
        See https://github.com/nmslib for more api details
    """
    def __init__(self, m=15, efc=100, num_threads=4, metric='cosinesimil'):
        self.m = m
        self.efc = efc
        self.num_threads = num_threads
        self.metric = metric
        self.index = None

    def _initialize_index(self, embeddings):
        index = nmslib.init(method='hnsw', space=self.metric)
        index.addDataPointBatch(embeddings)
        return index

    def _calculate_index(self,index):
        index_time_params = {'M': self.m,
                            'indexThreadQty': self.num_threads,
                            'efConstruction': self.efc}

        index.createIndex(index_time_params)
        return index

    def _load_index(self,index,index_path):
        index.loadIndex(index_path)
        return index

    def fit(self, path, index_path=None):
        embeddings,paths = self._gather_embeddings(path)
        index = self._initialize_index(embeddings)
        if index_path is None:
            index = self._calculate_index(index)
        else:
            index = self._load_index(index_path)
        index.setQueryTimeParams({'efSearch': self.efc})
        return index, paths


    def _gather_embeddings(self,path):
        files = Path(path).glob("*.pickle")
        embeddings, paths = [], []
        for f in files:
            ledger = self._read_pickle(str(f))
            embeddings.append(ledger['embedding'][None,...])
            paths.append(ledger['path'])
        return np.concatenate(embeddings,0),paths


    @staticmethod
    def _read_pickle(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)
