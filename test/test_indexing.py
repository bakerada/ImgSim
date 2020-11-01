from simtool.indexing import Indexer
import pytest

@pytest.fixture()
def indexer():
	return Indexer()

class TestIndexer:
	def test_gather_embeddings(self, indexer):
		path = 'test/data/embeddings'
		embeddings,paths = indexer._gather_embeddings(path)
		assert embeddings.shape[0] == 2
	def test_fit(self,indexer):
		path = 'test/data/embeddings'
		index, paths = indexer.fit(path)
