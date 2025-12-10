# Learning Mesh-Based Simulation with Graph Networks

This repository contains PyTorch implementations of [meshgraphnets](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets) for flow around circular cylinder problem on the basic of PyG (pytorch geometric).  

The original paper can be found as following:

 [Pfaff T, Fortunato M, Sanchez-Gonzalez A, et al. Learning mesh-based simulation with graph networks[J].](https://arxiv.org/pdf/2010.03409) International Conference on Learning Representations (ICLR), 2021.

Some code of this repository refer to [Differentiable Physics-informed Graph Networks](https://github.com/sungyongs/dpgn).


## Authors
-----------------------
  - Jiang
  - Zhang
  - Chu
  - Qian
  - Li
  - Wang


## Requirements
----------------------

``` bash
pip install -r requirements.txt
```

## Sample usage
-----------------------
- Download the dataset using:
```bash

aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/train.tfrecord -d data

aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/test.tfrecord -d data

aria2c -x 8 -s 8 https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/valid.tfrecord -d data
```
- Parse the downloaded dataset using the tool [parse_tfrecord.py](./parse_tfrecord.py), and the parsed files will be saved at `data` folder. Notice that the tensorflow version should lower than 1.15.0.
- train the model by run `python train.py`.
- For test, run `rollout.py`, and the result pickle file will be saved at result folder, the you can run the [render_results.py](./render_results.py) to generate result videos that can be saved at videos folder.
  
## Demos
-----------------------
- Here are some examples, trained on `cylinder_flow` dataset.
<img src="videos/0.gif" width="700" height="400"/>
<img src="videos/1.gif" width="700" height="400"/>


- In addition, we use simulation software to generate new training data. The test results on our data are as following:
<img src="videos/2.gif" width="700" height="400"/>
<img src="videos/3.gif" width="700" height="400"/>

## Contact me

:email: [jianglx@whu.edu.cn](jianglx@whu.edu.cn)
