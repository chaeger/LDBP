# LDBP: Learned Digital Backpropagation

## Getting Started

The code is based on [TensorFlow](https://www.tensorflow.org/) 1.13.1 and may
not work properly with other (older or newer) versions. It is recommended to create a
dedicated conda environment using the YAML file in the folder `conda` as
follows: 

```console
(base)~$ conda env create -f ldbp_env.yml
(base)~$ conda activate ldbp_env
```

Afterwards, it should be possible to run the provided jobscripts in the folder `ldbp`. For example: 

```console
(ldbp_env)~$ ./jobscript_isit
```

To train for different scenarios, most of the parameters and training options are set in a configuration file located in the folder `config`.

<!---
## Data Sets

```console
(ldbp_env)~$ htop
```
<p align="center"> 
<img src="htop.jpg">
</p>

```console
(ldbp_env)~$ watch nivida-smi
```
<p align="center"> 
<img src="nvidia.jpg">
</p>
-->

<!---
## Configurations

-->

## Additional Information

This repository is based on joint work with [Henry D.
Pfister](http://pfister.ee.duke.edu).
If you decide to use the source code for your research, please make sure
to cite our paper(s):

* C. Häger and H. D. Pfister, "[Nonlinear Interference Mitigation via Deep Neural Networks](https://arxiv.org/abs/1710.06234)", in Proc. *Optical Fiber Communication Conf. (OFC)*, San Diego, CA, March 2018

* C. Häger and H. D. Pfister, "[Deep Learning of the Nonlinear Schrödinger Equation in Fiber-Optic Communication](https://arxiv.org/abs/1804.02799)", In Proc. *IEEE Int. Symp. on Information Theory (ISIT)*, Vail, CO, June 2018

