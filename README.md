# image Similarity

The goal of this repo is to create a method for image retrieval.  The method utilizes a lightweight slightly modified ResNet architecture to perform image classification.  Before the final linear layer in the model architecture is a linear layer that creates an embedding vector for each image.  The method then uses 

#### Setup
To use this repository this path to this repo must be set in pythonpath
```
export PYTHONPATH=<path_to_repo>:$PYTHONPATH
```

#### Prep Data
To create the train / test / eval splits

```
python preprocessing/prep_data.py --rootdir <raw_datapath> --savedir <path_to_storesplits> --test_ratio 0.2 --eval_ratio 0.05
```

#### Training a Model
To train a model the config/train.json should be updated to reflect where the partition data is stored.  Other hyperparameters of the model can be tuned in this file as well.  Once the config/train.json file has been updated, the model can be trained by the following command:

```
python simtool/train.py
```
The model shoudl converge after about 30 minutes on a single GPU (p40 used for this exp)

#### Extract Embeddings
Once the model has converged the embeddings from the training data needs to be extracted.  These embeddings will be used to fit an approximate nearest neighbors method.  The embeddings can be extracted by running :

```
python simtool.extract_embeddings.py --datadir <path_to_splits> --savedir <embedding_path
```

This script requires the embedding_path parameter in config/deploy.json to be set to where the embeddings should be saved.  The script will extract the embeddings from the files specified by the traindir parameter in the config/train.json file.

#### Tests
To run the unit tests, and additional environment parameter must be set

```
export DATADIR=<raw_datapath>
```

Then simply run
```
pytest
```

#### Notebooks
The notebooks directory stores several jupyter notebooks to perform analysis
* The ApproximateNearestNeighbors notebook contains methods to fit the ANN method and compare it to a brute force
* The EmbeddingExporlation notebook performs T-SNE embedding and visualize to visualize the embeddings
* The ModelServing notebook visualizes the results of the topk searches for images

#### Deployment 
The ModelServer class can be deployed with seldon-core or on any kubernetes cluster.  As an example a small Amazon EKS cluter was spun up:

```
eksctl create cluster \
--name geo \
--version 1.13 \
--nodegroup-name standard-workers \
--node-type t3.nano \
--nodes 4 \
--nodes-min 1 \
--nodes-max 20 \
--node-ami auto
```
Then the deployment can be created by applying the simtool/deploy-app.yaml

```
kubectl apply -f simtool/deploy-app.yaml
```

An example request is 

```
curl http://<external_ip>:5000/predict \
    --request POST \
    --header "Content-Type: application/json" \
    --data '{"path": "/app/test/data/KO5OR.jpg","k": 2}'
```

where you can get the external ip with the command

```
kubectl get services
```
