# Efficient Hyperparameter Optimization in Deep Learning Using a Variable Length Genetic Algorithm

This is the code base for the paper [Efficient Hyperparameter Optimization in Deep Learning Using a Variable Length Genetic Algorithm](https://arxiv.org/abs/2006.12703v1) by Xueli Xiao, Ming Yan, Sunitha Basodi, Chunyan Ji, Yi Pan.

## Abstract
Convolutional Neural Networks (CNN) have gained great success in many artificial intelligence tasks. However, finding a good set of hyperparameters for a CNN remains a challenging task. It usually takes an expert with deep knowledge, and trials and errors. Genetic algorithms have been used in hyperparameter optimizations. However, traditional genetic algorithms with fixed-length chromosomes may not be a good fit for optimizing deep learning hyperparameters, because deep learning models have variable number of hyperparameters depending on the model depth. As the depth increases, the number of hyperparameters grows exponentially, and searching becomes exponentially harder. It is important to have an efficient algorithm that can find a good model in reasonable time. In this article, we propose to use a variable length genetic algorithm (GA) to systematically and automatically tune the hyperparameters of a CNN to improve its performance. Experimental results show that our algorithm can find good CNN hyperparameters efficiently. It is clear from our experiments that if more time is spent on optimizing the hyperparameters, better results could be achieved. Theoretically, if we had unlimited time and CPU power, we could find the optimized hyperparameters and achieve the best results in the future.

## Code structure
The code is written in `tf.keras`. The repository has the following notebooks:

1. **organism.ipynb** - This notebook deals with the code to build a single individual. The individual can be from any generation and from any phase.
2. **generation.ipynb** - This notebook deals with the code of building the problem of genetic algorithm. It run like a scripts which in turn generates specified number of organisms in a generation and then evolved the organisms. The variable genetic algorigthm also help the organisms evolve and increase their depth.

## WandB integration
This code has an integration with [weights and biases](https://wandb.ai). This helps in visualizing the metrics of the problem better and at run time. The code is also supported with a [wandb report](https://wandb.ai/authors/vlga/reports/Hyperparameter-optimization-with-Variable-Length-Genetic-Algorithm--VmlldzoyODA1Nzc).
