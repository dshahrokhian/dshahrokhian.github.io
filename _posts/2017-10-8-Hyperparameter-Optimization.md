---
layout: post
title: Hyperparameter Optimization in 10 minutes
published: true
---
In contrast to learned parameters, hyper-parameters cannot be directly learned from the training process. In the context of ML, hyper-parameter setting can be seen as model selection. Examples of hyper-parameters in ML models include:

**Learning rate** $\epsilon_t$ determines how quickly the gradient updates follow the direction of the gradient. If the learning rate is too small, the model will converge slowly and might prematurely end in a local minima that is far from optimal. If the learning rate is too large, the model will diverge by overshooting minimas. 

Furthermore, the learning rate is typically decreased over time. One approach is to set $\epsilon_t = \epsilon_0$ when $t < \tau$ after which we linearly decrease the learning rate by $\epsilon_t = \frac{\epsilon_0}{\tau^\alpha}$. Notice that $\tau$ becomes an additional hyper-parameter of the network.

**Loss function** can be seen as an energy function that is used to compare the network's output prediction with the ground truth during training. The choosing of the loss function can affect the accuracy of the model and the speed at which the network learns.

**Number of training iterations** or "epochs" determine how many times the algorithm trains on each sample of the data set.

**Mini-Batch size** determines the number of samples that are taken simultaneously on each training-step of the network. Large values increase the training speed while sacrificing learning per-step. Low values make the network learn really slow.

**Momentum** $\beta$ is a coefficient that determines how fast the old examples get down-weighted in the moving average, diminishing the fluctuations in weight changes over consecutive iterations.

**Number of Hidden Units** play a huge role in the generalization of the model. Larger than optimal values do not usually decrease the performance of the network, but it can heavily increase the amount of data necessary for training.

**Regularization coefficient** $\alpha$ acts as a penalty for model complexity. This complexity has nothing to do with the number of neurons and layers of the network, but with the learned weights. In general, weights close to 0 or 1 experience greater generality and avoid over-fitting the training data. There are various techniques to achieve this, such as [L1, L2](http://enhancedatascience.com/2017/07/04/machine-learning-explained-regularization/) and [Drop-out](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf).

# Manual Approaches

Hyper-parameter optimization is difficult, and there has been a lot of research done into trying to solve this problem. 

For many years, professionals have been basing their approach in past experience in the problem, tuning hyper-parameters by hand or using a specific range of values for each one, which is known as **Grid Search**. One could argue that this type of solution goes against the main objective of any ML task, which is levering the overall solution of the problem to be inferred by the algorithm through data. This empirical approach is computationally efficient, since it significantly reduces the search space. Nevertheless, the primary downside is the fact that this task can become tedious and probably non-practical for individuals without a deep knowledge in the domain of the problem. On the other end of the spectrum, one could use some sort of brute force in trying to find decent values. In practice, this solution is not feasible since the search space is enormous compared to the computational resources available at hand.

A technique that tries to mix both previous approaches is what is known as **Random Search**. This technique is empirically and theoretically proven to be more efficient than grid search and manual search to configure neural networks. The main idea behind it is the fact that, for most data sets, only a few hyper-parameters really matter, but that different hyper-parameters are important in different data sets. Instead of searching over all possible combinations, random search only evaluates a random sample of combinations. As shown in the paper, trying 60 random points samples from a grid seems to be a good approximation to what would be achieved by searching the entire space, and makes the solution way cheaper than grid search. The probabilistic explanation of picking that precise number is because the maximum of 60 random observations lies within the top 5\% of the true maximum with 95 \% probability. Each random draw has 5\% probability of landing in that interval, and if we draw n points independently, then the probability that all of them miss the desired interval is $(1-0.05)^n$. Thus:

\[1 - (1 - 0.05)^n > 0.95\]
\[n \geq 60\]

Yoshua Bengio explains in [his publication](https://arxiv.org/pdf/1206.5533.pdf) the practical recommendations for training deep architectures. 

# Automatic Approach: Bayesian Global Optimization

There is particular interest in automatic approaches that can optimize the hyper-parameters to the problem at hand. A good choice for achieving this is **Bayesian Optimization**, which has been shown to outperform other state-of-the-art techniques.

In recent years, Jasper \textit{et al.} explain [how to practically apply Bayesian Optimization methods for performing hyper-parameter tuning](https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf). This technique has been shown to be highly effective in different areas of machine learning. The results show significant speed-up with respect to other methods, and surpassed the state of the art in data sets such as CIFAR-10.

Optimization techniques tend to have the final goal of minimizing a given function $f(x)$, but what makes Bayesian optimization different is the fact that it constructs a probabilistic model of $f(x)$ while taking into account the uncertainty. The optimization usually works by assuming an unknown function sampled from a Gaussian process and maintains a posterior distribution of this function as observations are made.

A Gaussian process (GP) is a prior distribution of functions in the form $f:X \rightarrow \mathbb{R}$. A GP is defined by the property that any finite set of $N$ points $\{x_n \in X\}^{N}_{n=1}$ induces a multivariate Gaussian distribution on $\mathbb{R}^N$. These models assume that similar inputs give similar outputs. In the case of hyper-parameter optimization, we pick a setting of hyper-parameters such that the improvement with respect to the best setting seen so far is big. The process can be formalized in a set of steps, and understood easier with a single hyper-parameter.

![Bayesian Global Optimization](http://raw.githubusercontent.com/dshahrokhian/dshahrokhian.github.io/master/_posts/2017-10-8-Hyperparameter-Optimization/bayesian.png)

The plots show the mean and confidence intervals estimated with a probabilistic model of the objective function. Although the objective function is shown, in practice, it is unknown. The plots also show the acquisition functions in the lower shaded plots. The acquisition is high where the model predicts a high objective (exploitation) and where the prediction uncertainty is high (exploration). Note that the area on the far left remains unsampled, as while it has high uncertainty, it is correctly predicted to offer little improvement over the highest observation 

First, we define an objective function and an acquisition function. The objective function $f:X \rightarrow \mathbb{R}$ will react to the hyper-parameter setting and it is considered to be Gaussian distributed. The acquisition function ${a:X \rightarrow \mathbb{R}^{+}}$ determines the strategy to maximize the probability of improving over the best current value. Expected Improvement (EI) has been shown to be better-behaved than other metrics.

Second, we sample points in the objective function and update the fit or posterior of the GP. As it can be seen in the Figure, the variance of the GP decreases around the known points, acting as the uncertainty measurement. On the other hand, the EI decreases in those points.

Third, we find the point with the highest EI. We apply the objective function and update the GP, repeating the process with certain convergence tolerance and then returning the best solution.

