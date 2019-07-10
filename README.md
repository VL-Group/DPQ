# Deep Progressive Quantization

This is the python implementation of **Beyond Product Quantization: Deep Progressive Quantization for Image Retrieval** (IJCAI-19 accepted). Please refer to [paper link](http://arxiv.org/abs/1906.06698) for detailed infomation. The python scripts include following experiments:

- Training a model and Encoding a database [main.py](./main.py)
- Evaluating AQD mAP on query set [Eval.py](./Eval.py)
- Performing tSNE transform on CIFAR-10 [tSNE.py](./tSNE.py)
- Evaluating classification accuracy [Accuracy.py](./Accuracy.py)
- Checking time consumption [Time.py](./Time.py)

## Preliminaries

The scripts require following package:
- tensorflow (>= 1.4.1, either cpu or gpu version)
- numpy (>=1.14.3)
- scikit-learn (>=0.19.1)
- opencv-python (>=3.4)

The scripts have been tested on a machine equipped with:
- CPU: Intel(R) Xeon(R) CPU E5-2620 V4
- GPU: NVIDIA Titan X Pascal
- RAM: 128 GB (Please note that we didn't optimize memory usage, so the memory consumption would be very large)
- OS: Ubuntu 16.04 LTS

Please download AlexNet pretrianed model from ()()() and place it into `./data/models/`

### Dataset Preparation

CIFAR-10:

- Training set: 5000
- Query set: 1000
- database: 54000

You should split CIFAR-10 by yourself. [Download it](https://www.cs.toronto.edu/~kriz/cifar.html) and split it as the paper described.

---

NUS-WIDE:

- Training set: 10000
- Query set: 5000
- database: 190834

Please [download the dataset](https://github.com/lhmRyan/deep-supervised-hashing-DSH/issues/8#issuecomment-314314765) and put it into a specific directory. Then you should modify the prefixs of all paths in [/data/nus21](./data/nus21)

---

Imagenet:

- Training set: 10000
- Query set: 5000
- database: 128564

Please download the dataset (ILSVRC >= 2012) and put it into a specific directory. Then you should modify the prefixs of all paths in [/data/imagenet](./data/imagenet)

## Run

Firstly, list all the parameter of scripts:

    python main.py --help

    usage: main.py [-h] [--Dataset DATASET] [--Mode MODE] [--BitLength BITLENGTH]
               [--ClassNum CLASSNUM] [--K K] [--PrintEvery PRINTEVERY]
               [--LearningRate LEARNINGRATE] [--Epoch EPOCH]
               [--BatchSize BATCHSIZE] [--Device DEVICE] [--UseGPU [USEGPU]]
               [--noUseGPU] [--SaveModel [SAVEMODEL]] [--noSaveModel] [--R R]
               [--Lambda LAMBDA] [--Tau TAU] [--Mu MU] [--Nu NU]

    optional arguments:
    -h, --help            show this help message and exit
    --Dataset DATASET     The preferred dataset, 'CIFAR', 'NUS' or 'Imagenet'
    --Mode MODE           'train' or 'eval'
    --BitLength BITLENGTH
                            Binary code length
    --ClassNum CLASSNUM   Label num of dataset
    --K K                 The centroids number of a codebook
    --PrintEvery PRINTEVERY
                            Print every ? iterations
    --LearningRate LEARNINGRATE
                            Init learning rate
    --Epoch EPOCH         Total epoches
    --BatchSize BATCHSIZE
                            Batch size
    --Device DEVICE       GPU device ID
    --UseGPU [USEGPU]     Use /device:GPU or /cpu
    --noUseGPU
    --SaveModel [SAVEMODEL]
                            Save model at every epoch done
    --noSaveModel
    --R R                 mAP@R, -1 for all
    --Lambda LAMBDA       Lambda, decribed in paper
    --Tau TAU             Tau, decribed in paper
    --Mu MU               Mu, decribed in paper
    --Nu NU               Nu, decribed in paper

To perform a training:

    python main.py --Dataset='CIFAR' --ClassNum=10 --LearningRate=0.001 --Device=0

To perform an evaluation:

    python Eval.py --Dataset='CIFAR' --ClassNum=10 --Device=0 --R=-1

## Citations

Please use the following bibtex to cite our papers:

```
@inproceedings{DPQ,
  title={Beyond Product Quantization: Deep Progressive Quantization for Image Retrieval},
  author={Gao, Lianli and Zhu, Xiaosu and Song, Jingkuan and Zhao, Zhou and Shen, Heng Tao},
  booktitle={Proceedings of the 2019 International Joint Conferences on Artifical Intelligence (IJCAI)},
  year={2019}
}
```
```
@inproceedings{DRQ,
  title={Deep Recurrent Quantization for Generating Sequential Binary Codes},
  author={Song, Jingkuan and Zhu, Xiaosu and Gao, Lianli and Xu, Xin-Shun and Liu, Wu and Shen, Heng Tao},
  booktitle={Proceedings of the 2019 International Joint Conferences on Artifical Intelligence (IJCAI)},
  year={2019}
}
```
