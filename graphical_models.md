# Graphical Models

My notes on Chapter 8 of "Pattern Recognition and Machine Learning" by Bishop.
I pretty much exactly follow the chapter, but with paraphrasing and a little more brevity.
These were made for my benefit only, and you're probably better off reading the actual chapter!

## Bayesian Networks

### Introduction

We can represent causal models with *directed, acyclic graphs* (DAGs).
Consider a scenario where there are three random variables $a$, $b$ and $c$.
We can write the joint distribution of these as
$$p(a, b, c) = p(c | a, b)p(a, b).$$
We can then further factorise this as
$$p(a, b, c) = p(c | a, b)p(b | a)p(a).$$
We translate this into a graph by representing the conditional distribution of each variable with a *node*.
Then, for each conditional distribution we add *directed links* (arrows) corresponding to the variables upon which the distribution is conditioned.
For the simple example above, the graph is given in Figure \ref{fig1}.

![A simple, *fully connected* graph. The graph is fully connected because there is a link between each pair of nodes.\label{fig1}](figures/fig_8p1.pdf)

In general, the rule for extracting a joint distribution from a graph with $K$ nodes is
$$p(\boldsymbol{x}) = \prod \limits_{k=1}{K} p(x_k | \mathrm{pa}_k),$$
where $\mathrm{pa}_k$ represents the set of *parents* of $x_k$.
This equation expresses the *factorisation* properties of the graph.
We call the node at which an arrow points the *child* of the node where the arrow originates, which is the *parent*.

![A more interesting graph.\label{fig2}](figures/fig_8p2.pdf)

Graphs are more interesting when they are not fully connected.
Figure \ref{fig2} is an example of a more interesting graph.
The joint distribution of this graph is
$$p(x_1)p(x_2)p(x_3)p(x_4|x_1,x_2,x_3)p(x_5|x_1,x_3)p(x_6|x_4)p(x_7|x_4,x_5).$$

### Linear regression example

Take a linear model that is expressed by

$$\boldsymbol{t} \sim \mathrm{N}\left(\boldsymbol{w} . \boldsymbol{x}, \sigma^2\right),$$
$$\boldsymbol{w} \sim \mathrm{N}\left(0, \alpha\right).$$

where there are covariates $x_n$ and observed data $t_n$, and weights $\boldsymbol{w}$.
The prior on the weights is a normal with variance $\alpha$.
The graph of this model is shown in Figure \ref{fig3}.
In this graph, we have introduced three new pieces of notation:

* Where there are $N$ nodes in the graph with the same parents and children, we summarise the nodes with a single effective node, and surround that node by a *plate*.
If there are dependencies within a given plate (as there are in this graph), then the implication is that e.g. $t_n$ depends on $x_n$, but not on $x_m$, where $m \neq n$.
In this case, the $t_n$ and $x_n$ are surrounded by a plate, and the plate is labelled with $N$ to inform us that there are $N$ such nodes.  
* Nodes that are represented by a dot instead of a circle are *fixed* in value.
The implication in this particular graph is that the parameter $\sigma^2$ and the hyperparameter $\alpha$ are fixed at the outset, and are not inferred.
Furthermore, because the $x_n$ are covariates (we do not model them as random variables), they are also fixed.
When variables are fixed, we do not include them when we produce the joint distribution of the graph (because we only ever condition on them, since they are not random variables).  
* If a variable is *observed*, then the plate is *shaded*, as is the case for the $t_n$.

![The graph of a linear model.\label{fig3}](figures/fig_8p5.pdf)

When we fit a linear model like this one, we want to use it to predict values of $t$ for new observations $\hat x$.
Figure \ref{fig4} expands the existing graph to include the dependencies of the predictions $\hat t$ and the new observations $\hat x$.
The explicit joint distribution of the new graph is
$$p(\hat t, \boldsymbol{t}, \boldsymbol{w} | \hat x, \boldsymbol{x}, \alpha, \sigma^2) = \left[\prod\limits_{n=1}^N p(t_n | x_n, \boldsymbol{w}, \sigma^2) \right]p(\boldsymbol{w} | \alpha)p(\hat t | \hat x, \boldsymbol{w}, \sigma^2).$$
We are of course not very interested in evaluating this full joint distribution, but rather the predictive distribution for $\hat t$, which can be computed by integrating the above equation over $\boldsymbol{w}$ and by regarding $\boldsymbol{t}$ as fixed to the values in the training set.

$$p(\hat t | \hat x, \boldsymbol{x}, \boldsymbol{t}, \alpha, \sigma^2) \propto \int \mathrm{d}\boldsymbol{w}\,p(\hat t, \boldsymbol{t}, \boldsymbol{w} | \hat x, \boldsymbol{x}, \alpha, \sigma^2).$$
To make this distribution properly normalised, we have to integrate the resulting distribution over $\hat t$ (hence the proportionality symbol as opposed to equality).
We generally won't mind that the distribution isn't normalised if we are using MCMC methods to sample from the model.

![The linear model graph extended to show the dependencies of new predictions.\label{fig4}](figures/fig_8p7.pdf)

### Sampling from a graphical model

Suppose we have a graph where each node is a random variable $x_n$.
We label the random variables such that each node has a higher index than its parents, e.g. $x_1$ can have no parents, and $x_2$ can only have $x_1$ as a parent (although it may not), and $x_3$ could have $x_1$ and $x_2$ as parents, etc.
Given this ordering, we can sample from the model in a similar fashion to Gibbs sampling: we first draw a single sample $\hat x_1$ from $p(x_1)$, then a single sample from the conditional distribution $p(x_2 | \mathrm{pa}_2)$ (where $\mathrm{pa}_2 \in (x_1, \mathrm{none})$), and so on.
It is identical to Gibbs sampling, except that we must be careful to sample from the graph in the correct order, so that there are up-to-date values of the parents of the node $x_n$ available when we sample from its conditional distribution.
We can then obtain valid samples from any of the joint distributions, e.g. $p(x_1, x_100)$, by simply discarding all samples except those from the nodes of interest (this is identical to the way we obtain marginal distributions from MCMC samples).

This leads to an interpretation of graphical models as *generative* or *causal* models.
The edges in the graph represent causality: a parent node (perhaps partially) "causes" its children.
Not all models are fully generative, however.
For example, the linear model above is not generative because the covariates $\boldsymbol{x}$ are not random variables, but they are required to produce the observations $\boldsymbol{t}$.

I'm not really sure that I understand this properly: why can't we sample from the random variables and ignore the fixed ones?
In the linear model example, one can still produce synthetic data sets $\boldsymbol{t}$ under the presumption that the covariates are always the same.
Maybe the author is drawing a distinction between modelling *all of the data* and *some of the data* (or between inferential and predictive modelling, or something).

### Discrete variables

![Top: the case when the two random variables are causally linked, and there are $K^2 - 1$ different model parameters. Bottom: the case when the two random variables are not causally linked, and there are $2(K - 1)$ parameters. \label{fig5}](figures/fig_8p9.pdf)

Consider a random variable $\boldsymbol{x}$ that can take one of $K$ different values (or states).
If we write $\boldsymbol{x}_1$ in the one-of-$K$ representation (i.e., one hot encoding, where the variable is represented as a length $K$ vector with zeros everywhere except for the current state, which has value 1), then the probability distribution is
$$\mathrm{Pr}(\boldsymbol{x}_1 | \boldsymbol{\mu}_1) = \prod\limits_{k=1}^K \mu_{1k}^{x_{1k}},$$
where $\mu_{1k}$ is the probability of the RV being in state $k$.
We need only specify $K-1$ of the parameters because these probabilities must sum to unity.

Now we introduce a second RV $\boldsymbol{x}_2$.
We presume that the two variable are correlated, such that we must now produce a matrix of probabilities $\mu_{kl}$, which is interpreted as the probability that $\boldsymbol{x}_1$ is in state $k$ and $\boldsymbol{x}_2$ is in state $l$.
The probability distribution over states is now
$$\mathrm{Pr}(\boldsymbol{x}_1, \boldsymbol{x}_2 | \boldsymbol{\mu}) = \prod\limits_{l=1}^{K}\prod\limits_{k=1}^{K} \mu_{lk}^{x_{1k}x_{2l}}.$$
We again have a constraint that means that the number of independent components of $\boldsymbol{\mu}$ is $K^2 - 1$.
For $M$ RVs, the number of independent components is $K^M - 1$.

We can imagine this graphically.
Figure \ref{fig5} shows two possible graphs for $\boldsymbol{x}_1$ and $\boldsymbol{x}_2$.
The first graph has joint distribution
$$p(\boldsymbol{x}_1)p(\boldsymbol{x}_2 | \boldsymbol{x}_1).$$
The first factor has $K-1$ parameters (the $K$ states of $\boldsymbol{x}_1$) and the second factor has $K-1$ parameters corresponding to $\boldsymbol{x}_2$ for each of the $K$ states of $\boldsymbol{x}_1$.
The total number of parameters is thus $K - 1 + K(K - 1) = K^2 - 1$.
The second graph has joint distribution
$$p(\boldsymbol{x}_1)p(\boldsymbol{x}_2),$$
where each factor has $K-1$ associated parameters, giving $2(K - 1)$ parameters in total.

If we have $M$ discrete variables, then we can imagine different configurations.
For example, the fully connected graph has $K^M - 1$ parameters.
But if the graph is configured like a chain (every node has two connections: one to their parent and one to their child, except for the first and last nodes in the graph, which have one connection to either a parent or a child), then there are instead $K - 1$ parameters for the first node, and then $K(K - 1)$ parameters for each of the subsequent nodes, giving $K - 1 + (M - 1)K(K - 1)$ parameters altogether.
Thus the graphical representation allows us to produce a wide variety of different configurations of discrete random variables.
We could reduce the number of parameters further by *sharing* (or *tying*) parameters.
For example, we could specify that $p(x_i | x_{i -1})$ is governed by the same set of parameters, irrespective of $i$.

Continuing with the example of a chain configuration, we can turn the graph of random variables into a Bayesian model (it is not currently interpreted as such because there are no prior distributions of the parameters $\boldsymbol{\mu}$) by associating each node with another parent node that represents the prior on $\boldsymbol{\mu}$.
Figures \ref{fig6} and \ref{fig7} show two versions of a Bayesian model: one where each of the conditional distributions has its own parameters, and another where parameters are shared between conditional distributions.

![A version of the first graph in Figure \ref{fig5} where priors on the multinomial parameters are specified. \label{fig6}](figures/fig_8p11.pdf)

![A version of the graph in Figure \ref{fig6} where the multinomial parameters are shared between the conditional distributions $p(x_i | x_{i -1})$. \label{fig7}](figures/fig_8p12.pdf)

Another way to control the number of parameters is to use parametric models for the conditional distributions, as opposed to the full matrices $\mu$.
Consider a situation where there are $M$ binary variables $x_i$ and a further binary variable $y$.
Figure \ref{fig8} shows the graph we will consider.
There are $M$ parameters controlling the states of the $x_i$, and notionally there are $2^M$ parameters controlling the conditional distribution of $y$ (one parameter for $y=1$ for each of the $2^M$ settings of the parent variables).
If $M$ is anything even mildly large (100 parents would give $> 10^{30}$ parameters), then fully parameterising the model in this way is madness.
Instead, one could use a logistic regression model, so that
$$p(y=1 | x_1, \cdots , x_M) = \sigma\left(w_0 + \sum\limits_{i=1}^{M} w_i x_i\right) = \sigma(\boldsymbol{w}^T\boldsymbol{x}),$$
where $\sigma(x) = \left(1 + \exp -x\right)^{-1}$ is the sigmoid function.
We have now reduced the number of parameters in the model from $2^M$ to $M + 1$.

![A graph of $M$ parents and a single child. \label{fig8}](figures/fig_8p13.pdf)

### Linear-Gaussian models

This example shows how a multivariate Gaussian distribution (mvg) can be expressed as a dag corresponding to a linear-Gaussian model over the component variables.
Consider a dag over $D$ variables in which node $i$ represents a single continuous random variable $x_i$ having a Gaussian distribution.
The mean of this distribution is taken to be a linear combination of the states of its parent nodes $\mathrm{pa}_i$ of node $i$
$$p(x_i | \mathrm{pa}_i) = \mathcal{N}\left(x_i \bigg| \sum\limits_{j\in\mathrm{pa}_i}w_{ij}x_j + b_i, v_i\right)$$
where $w_{ij}$ and $b_i$ are parameters governing the mean and $v_i$ is the variance of the conditional distribution for $x_i$.
The log of the joint distribution is the log of the product of the conditionals over all nodes in the graph

\begin{equation}
\begin{split}
\log p(\boldsymbol{x}) & = \log p(x_i | \mathrm{pa}_i) \\
 & = -\sum\limits_{i=1}^D \dfrac{1}{2v_i}\left(x_i - \sum\limits_{j\in\mathrm{pa}_i}w_{ij}x_j - b_i\right)^2 + \mathrm{const}.
\end{split}
\end{equation}

Because the log of the joint distribution is a quadratic in the components $\boldsymbol{x}$, the joint distribution is a Gaussian.
The mean and variance of the joint distribution can be calculated recursively.
We can write the variable at each node as
\begin{equation}x_i = \sum\limits_{j\in \mathrm{pa}_i} w_{ij}x_j + b_i + \sqrt{v_i} \epsilon_i,\label{eq:lingauss_1}\end{equation}
where $\epsilon_i$ is a new random variable associated with node $i$ that has zero mean and unit variance.
Taking the expectation of Equation (\ref{eq:lingauss_1}), we get
$$\mathbb{E}\left[x_i\right] = \sum\limits_{j\in \mathrm{pa}_i} w_{ij}\mathbb{E}\left[x_j\right] + b_i.$$
So, by starting with the lowest numbered node (remember the convention that $x_i$ can only have $x_{j<i}$ as parents), we can calculate the mean vector of the joint distribution: $\mathbb{E}[\boldsymbol{x}] = \left(\mathbb{E}[x_1, \dots, x_D]\right)^T$.
We can obtain the covariance matrix in the same way

\begin{equation}
\begin{split}
\mathrm{cov}\left[x_i, x_j\right] &= \mathbb{E}\left[\left(x_i - \mathbb{E}[x_i]\right)\left(x_j - \mathbb{E}[x_j]\right)\right] \\
&= \mathbb{E}\left[(x_i - \mathbb{E}[x_i])\left\{\sum\limits_{k\in \mathrm{pa}_j}w_{jk}(x_k - \mathbb{E}[x_k]) + \sqrt{v_j}\epsilon_j\right\}\right] \\
&= \sum\limits_{k\in \mathrm{pa}_j} w_{kj}\mathrm{cov}[x_i, x_k] + I_{ij}v_j.
\end{split}
\end{equation}

There are two limiting cases: the case when the graph has no connections, which means that the
