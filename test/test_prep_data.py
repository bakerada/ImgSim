from preprocessing.prep_data import *
import pytest
from pathlib import Path

class TestDataPrep:
	def test_get_classes(self):
		classes = get_classes(os.environ['DATADIR'])
		assert len(classes) == 6

	def test_get_split_counts(self):
		test_count, eval_count = get_split_counts(Path(os.environ['DATADIR']),
			                                           test_ratio= 0.2,
			                                           eval_ratio= 0.05)

		assert test_count == 1000.
		assert eval_count == 250.

	def test_sample_dir(self):
		samples = sample_dir(Path(os.environ['DATADIR']) / 'class_1',
                             test_count= 1000,
                             eval_count= 250)

		assert len(samples['test']) == 1000
		assert len(samples['eval']) == 250
