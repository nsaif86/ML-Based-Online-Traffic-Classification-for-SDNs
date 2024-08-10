This project represents the work in our paper submitted to [2022 IEEE 2nd Conference on Information Technology and Data Science (CITDS)](https://ieeexplore.ieee.org/xpl/conhome/9913999/proceeding) "ML-Based Online Traffic Classification for SDNs",  DOI: [10.1109/CITDS54976.2022.9914138](https://doi.org/10.1109/CITDS54976.2022.9914138), Authors: [Mohammed Nsaif](https://ieeexplore.ieee.org/author/37089577341); [Gergely Kovásznai](https://ieeexplore.ieee.org/author/37569885400); [Mohammed Abboosh](https://ieeexplore.ieee.org/author/37089568445); [Ali Malik](https://ieeexplore.ieee.org/author/37088446887); [Ruairí de Fréin](https://ieeexplore.ieee.org/author/37086819785)

### In this project, we provide:

- SDN application source codes
- Our Testbed configuration, deployment including scripts and used resources
- Model training and evaluation source codes
- Every above component will be described in this README as much as possible.

### Dependencies

#### OS: Ubuntu 20.04.x

>  If you are using Windows or other OS, you can install Ubuntu 20.04 as a Virtual Machine (VM) using Virtual Box. You can download the ISO installer from this [link.](https://www.ubuntu.com/download/desktop)

####  Ryu controller

> We use [Ryu](https://ryu-sdn.org/) to deploy our management and monitoring SDN application. Ryu can work in any OS Environment that supports Python 3. You can install Ryu as follows:

 ```sudo pip install ryu```

#### Mininet

> To simulate an SDN network, we use the popular framework [Mininet](http://mininet.org/). Mininet currently only works in Linux. In our project, we run mininet in an Ubuntu 20.04.2 LTS VM. To get mininet, you can simply download a compressed Mininet VM from [Mininet downloadpage](https://github.com/mininet/mininet/wiki/Mininet-VM-Images) or install it through apt: 
```
sudo apt update 
sudo apt install mininet
```
> or install natively from source:
```
git clone git://github.com/mininet/mininet
cd mininet
git tag  # list available versions
git checkout -b cs244-spring-2012-final  # or whatever version you wish to install
util/install.sh -a
```

#### Openvswitch Installation: 
A collection of guides detailing how to install Open vSwitch in a variety of different environments and using different configurations can find [here](https://docs.openvswitch.org/en/latest/intro/install/). However, for ubuntu as follows:

```
sudo apt-get install openvswitch-switch
```

#### Distributed Internet Traffic Generator
> D-ITG is a platform capable of producing traffic at the packet level accurately replicating appropriate stochastic processes for both IDT (Inter Departure Time) and PS (Packet Size) random variables (Exponential, Uniform, Cauchy, normal, Pareto, ...).
D-ITG supports both IPv4 and IPv6 traffic generation and it is capable of generating traffic at network, transport, and application layer.
Install and usage guidelines, you can be found [here](https://traffic.comics.unina.it/software/ITG/)

> Quique installation as follows:

```
sudo apt-get install D-ITG
```

#### Machine Learning packages

> In our project, we use [keras](https://keras.io/) and tensorflow framework to train our models, in addition, we also use several packages such as pandas, numpy, sklearn to preprocess data for training and matplotlib for result visualization. All ML source codes are written in jupyter-notebook. The packages can be installed by pip as follows:

```
python -m pip install -U pip
python -m pip install tensorflow, keras, numpy, pandas, scikit-learn, matplotlib, jupyter-notebook
```

> https://www.tensorflow.org/install/pip


### SDN Applications Usage

In this project, we created two Ryu applications, and the source codes are stored in the  [SDN-Classifier directory](https://github.com/nsaif86/SDN-Classifier). The first component, the classifier, is responsible for collecting and extracting the features and feeding the features of each flow into the training saved model it is the same APPa used in [project](https://github.com/a9khan/trafficclassifier) except that we used neural network mode classifier. The method is clearly described in our paper. The second one is the topology model

In our situation, we have omitted network management on network services, so a flow for IP packets simply consists of only two matching fields: Source and Destination MAC Address. This simplicity reduces the load on the controller and helps switch forward packets faster.

To run the applications, copy the two Python programs into the ryu.app folder of the Ryu directory and open two terminal windows, execute the following command:

In the first terminal
```
sudo python3  traffic_classifier_python3.py neural_network

```
In the second terminal
```
sudo python3 topo.py
```
Then, you can generate traffic (voice, DDoS, etc.) using D-ITG inside Mininet and observe the results in the first terminal.

See this [video](https://youtu.be/3vOvAD6iaac), which includes a practical example of running the code at the end.. Please, if you find the code useful, cite our work.

@inproceedings{nsaif2022ml,
  title={ML-based online traffic classification for SDNs},
  author={Nsaif, Mohammed and Kov{\'a}sznai, Gergely and Abboosh, Mohammed and Malik, Ali and de Fr{\'e}in, Ruair{\'\i}},
  booktitle={2022 IEEE 2nd Conference on Information Technology and Data Science (CITDS)},
  pages={217--222},
  year={2022},
  organization={IEEE}
}
