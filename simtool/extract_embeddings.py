from simtool.dataloader import *
from utils.helpers import read_config, to_numpy
from simtool.model import RockClassifier
import argparse
import torch
from collections import OrderedDict
import torchvision.transforms.functional as F
import os
import numpy as np
from pathlib import Path
import pickle
import torch.nn as nn


model_map = {
    "resnet": RockClassifier
}

class EmbeddingExtractor:
    """ A class to extract embeddings from the residual classification network
            Args:
                model_path (str): Location to model checkpoint
                model (str): Key for model_map to select model architecture
                batch_size (int): Number of images to pass through model at a time
                num_works (int): Processes for data loader
                batch_config (dict): Additional data loader arguments
                model_kwargs (kwargs): parameters needed to intialize network 
    """
    def __init__(self, model_path,
                      gpus=[], model='resnet',
                      batch_size=32, num_works=2,
                      batch_config={}, **model_kwargs):
        self.batch_size = batch_size
        self.batch_config = batch_config
        self. num_works = num_works
        self.model = self.load_model(model_path, model, gpus, **model_kwargs)
        self.use_cuda = next(self.model.parameters()).is_cuda


    def load_model(self,model_path, model, gpus, **kwargs):
        model = model_map[model](**kwargs)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if k[:7] == 'module.':
                name = k[7:]
            else:
                name = k
            state_dict[name] = v
        model.load_state_dict(state_dict)
        if (len(gpus) > 0) and (torch.cuda.is_available()):
            model = nn.DataParallel(model,device_ids = gpus)
        return model.eval()


    def _process_img(self,path):
        transformations = build_transforms(self.batch_config['transforms'])
        img = pil_loader(path)
        for t in transformations:
            img = t(img)
        img = img.unsqueeze(0).float()

        if self.use_cuda:
            img = img.cuda()
        pred, embedding = self.model(img)
        return np.argmax(to_numpy(pred.squeeze(0))), to_numpy(embedding.squeeze(0)), path

    def _process_dir(self,path):
        loader = create_dataloader(path, self.batch_config)
        predictions = []
        embeddings = []
        paths = [x[0] for x in loader.dataset.imgs]
        for batch_idx,data in enumerate(loader):
            img = data[0].float()
            if self.use_cuda:
                img = img.cuda()
            out, embedding = self.model(img)
            predictions.append(np.argmax(to_numpy(out),1))
            embeddings.append(to_numpy(embedding))

        return np.concatenate(predictions,0), np.concatenate(embeddings,0), paths


    def _determine_method(self, path, is_file=False):
        if os.path.isfile(path):
            method = self._process_img
            is_file = True
        else:
            method = self._process_dir
        return method, is_file

    @staticmethod
    def _dissect_path(path):
        p = Path(path)
        return p.parts[-2:]

    @staticmethod
    def _save_pickle(path,obj):
        with open(path, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _save_file(self, predictions, embeddings, path, save_dir):
        label, img = self._dissect_path(path)
        save_path = os.path.join(save_dir, label + '_' + img.split('.')[0].lower())
        data = {"prediction": predictions, "embedding": embeddings, "path":path}
        self._save_pickle(save_path + '.pickle',data)

    def _save_embeddings(self, predictions, embeddings, paths, save_dir, is_file=False):
        if is_file:
            self._save_file(predictions, embeddings, paths, save_dir)
        else:
            for i in range(predictions.shape[0]):
                self._save_file(predictions[i],embeddings[i],paths[i],save_dir)


    def extract_embeddings(self, path, save_dir=None):
        method, is_file = self._determine_method(path)
        predictions, embeddings, paths = method(path)
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            self._save_embeddings(predictions,embeddings,paths,save_dir, is_file)
        else:
            return predictions, embeddings, paths



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Extract Embeddings")
    parser.add_argument('--datadir', type=str, help="Path split data")
    parser.add_argument('--savedir', type=str, help="Path to where embeddings are saved")
    args = parser.parse_args()

    basedir = args.datadir
    save_dir = args.savedir
    folders = ['eval','train','test']
    config = read_config('config/deploy.json')
    extractor = EmbeddingExtractor(config['model_path'],
                                   gpus = config['gpus'],
                                   batch_config = config['data'],
                                   **config['model_args'])

    for f in folders:
        folder_path = os.path.join(basedir,f)
        save_folder = os.path.join(save_dir,f)
        extractor.extract_embeddings(folder_path,save_dir=save_folder)
        print("saved folder: {}".format(f))
