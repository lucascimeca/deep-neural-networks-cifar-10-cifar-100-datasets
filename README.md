# Deep Neural Networks - CIFAR-10/CIFAR-100 datasets

Author: Luca Scimeca

The code in the "/mlp" folder was adapted from the course "Machine Learning Practical" at the University of Edinburgh [School of Informatics](http://www.inf.ed.ac.uk) (http://www.inf.ed.ac.uk/teaching/courses/mlp/).
The remaining folders contain custom made code, built for the purpose of the practical. 

The code in this repository is split into:

  *  a Python package `mlp`, a [NumPy](http://www.numpy.org/) based neural network package designed specifically for the course.
  *  An 'experiment_set_1' folder containing the first set of experiments run for CIFAR 10 and CIFAR 100. The folder contains
	 Jupyter notebooks implementing the NNs described in the report (also contained in the folder) through tensor flow.
  *  An 'experiment_set_2' folder containing the second set of experiments run for CIFAR 10 and CIFAR 100. The folder contains
	 Python 3 files which implement the NNs described in the report (also contained in the folder) through tensor flow. 
	 This set of experiments focues on Multitask learning, and a series of personally developed methods to improve classification
	 through a combination of learning rules and the idea of Multitask Learning.
  *  '/data' folder should contain the datasets
  *  '/figures' folder containing the figures used in the resports.

## To see my implementation of Multitask Learning through the newly deviced learning methods see the python files in '/experiment_set_2'