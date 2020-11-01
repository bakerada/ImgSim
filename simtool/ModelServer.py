import os

class ModelServer:
	def __init__(self):
		self.object = os.environ['CONFIG_PATH']

	def predict(self, X, feature_names):
		return self.object