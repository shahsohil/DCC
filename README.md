# Deep Continuous Clustering #

## Introduction ##

This is a Pytorch implementation of the DCC algorithms presented in the following paper ([paper](http://arxiv.org/abs/1803.01449)):

Sohil Atul Shah and Vladlen Koltun. Deep Continuous Clustering.

If you use this code in your research, please cite our paper.
```
@article{shah2018DCC,
	author    = {Sohil Atul Shah and Vladlen Koltun},
	title     = {Deep Continuous Clustering},
	journal   = {arXiv:1803.01449},
	year      = {2018},
}
```

The source code and dataset are published under the MIT license. See [LICENSE](LICENSE) for details. In general, you can use the code for any purpose with proper attribution. If you do something interesting with the code, we'll be happy to know. Feel free to contact us.

## Requirement ##

* Python 2.7
* [Pytorch](http://pytorch.org/) >= v0.2.0
* [Tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch)

## Pretraining SDAE ##

##### Note: Please find required files and checkpoints for MNIST dataset shared [here](https://drive.google.com/drive/folders/10DjPtVRHgZcM-dshm4MuyB5DmxpfG_hV?usp=sharing).

Please create new folder for each dataset under the [data](data) folder. Please follow the structure of [mnist](data/mnist) dataset. The training and the validation data for each dataset must be placed under their respective folder.

We have already provided [train](data/mnist/traindata.mat) and [test](data/mnist/testdata.mat) data files for MNIST dataset. For example, one can start pretraining of SDAE from console as follows:

```
$ python pretraining.py --data mnist --tensorboard --id 1 --niter 50000 --lr 10 --step 20000
```

Different settings for total iterations, learning rate and stepsize may be required for other datasets. Please find the details under the comment section inside the [pretraining](pytorch/pretraining.py) file.

## Extracting Pretrained Features ##

The features from the pretrained SDAE network are extracted as follows:

```
$ python extract_feature.py --data mnist --net checkpoint_4.pth.tar --features pretrained
```

By default, the model checkpoint for pretrained SDAE NW is stored under [results](data/mnist/results).

## Copying mkNN graph ##

The [copyGraph](pytorch/copyGraph.py) program is used to merge the preprocessed mkNN graph (using the code provided by [RCC](https://bitbucket.org/sohilas/robust-continuous-clustering/src)) and the extracted pretrained features. Note the mkNN graph is built on the original and not on the SDAE features.

```
$ python copyGraph.py --data mnist --graph pretrained.mat --features pretrained.pkl --out pretrained
```

The above command assumes that the graph is stored in the [pretrained.mat](data/mnist/pretrained.mat) file and the merged file is stored back to pretrained.mat file. 

##### [DCC](pytorch/DCC.py) searches for the file with name pretrained.mat. Hence please retain the name. #####

## Running Deep Continuous Clustering ##

Once the features are extracted and graph details merged, one can start training DCC algorithm. 

For sanity check, we have also provided a [pretrained.mat](data/mnist/pretrained.mat) and SDAE [model](data/mnist/results/checkpoint_4.pth.tar) files for the MNIST dataset located under the [data](data/mnist) folder. For example, one can run DCC on MNIST from console as follows:

```
$ python DCC.py --data mnist --net checkpoint_4.pth.tar --tensorboard --id 1
```

The other preprocessed graph files can be found in gdrive [folder](https://drive.google.com/drive/folders/1vN4IpmjJvRngaGkLSyKVsPaoGXL02mFf?usp=sharing) as provided by the RCC.

### Evaluation ###

Towards the end of run of DCC algorithm, i.e., once the stopping criterion is met, [DCC](pytorch/DCC.py) starts evaluating the cluster assignment for the total dataset. The evaluation output is logged into tensorboard logger. The penultimate evaluated output is reported in the paper.

##### Like RCC, the AMI definition followed here differs slightly from the default definition found in the sklearn package. To match the results listed in the paper, please modify it accordingly. #####

The tensorboard logs for both pretraining and DCC will be stored in the "runs/DCC" folder under [results](data/mnist/results/). The final embedded features 'U' and cluster assignment for each sample is saved in 'features.mat' file under [results](data/mnist/results/).  

### Creating input ###

The input file for SDAE pretraining, [traindata.mat](data/mnist/traindata.mat) and [testdata.mat](data/mnist/testdata.mat), stores the features of the 'N' data samples in a matrix format N x D. We followed 4:1 ratio to split train and validation data. The provided [make_data.py](pytorch/make_data.py) can be used to build training and validation data. The distinction of training and validation set is used only for the pretraining stage. For end-to-end training, there is no such distinction in unsupervised learning and hence all data has been used. 

To construct mkNN edge set and to create preprocessed input file, [pretrained.mat](data/mnist/pretrained.mat), from the raw feature file, use [edgeConstruction.py](https://bitbucket.org/sohilas/robust-continuous-clustering/src/0516c0e1c65027ca0ffa1f09e0aa3074b99dea80/Toolbox/edgeConstruction.py) released by RCC. Please follow the instruction therein. Note that mkNN graph is built on the complete dataset. For simplicity, code (post pretraining phase) follows the data ordering of \[trainset, testset\] to arrange the data. This should be consistent even with mkNN construction.
