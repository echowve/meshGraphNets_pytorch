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
-----------------------
  - h5py==3.6.0
  - matplotlib==3.4.3
  - numpy==1.21.1
  - opencv_python==4.5.4.58
  - Pillow==9.1.0
  - torch==1.9.0+cu111
  - torch_geometric==2.0.4
  - torch_scatter==2.0.8
  - tqdm==4.62.3

  ``` bash
  pip install -r requirements.txt
  ```

## Sample usage
-----------------------

- Download `cylinder_flow` dataset using the script https://github.com/deepmind/deepmind-research/blob/master/meshgraphnets/download_dataset.sh.

- Parse the downloaded dataset into `.h5` file using the tool [parse_tfrecord.py](./parse_tfrecord.py)
- Change the `dataset_dir` in [train.py](./train.py) to your `.h5` files.
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
