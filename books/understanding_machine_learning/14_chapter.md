# 14 Stochastic Gradient Descent

In stochastic gradient descent (SGD) we try to minimize
the risk function $L_D(w)$ directly using a gradient 
descent procedure. Gradient descent is an iterative
optimization procedure where at each step the solution
is improved by taking a step along the negative of the gradient. 
But, the function to optimize is not known since 
the true distribution is unknown. SGD circumvents this
by taking a step in a random direction as long as the
expected value of the direction is the negative of the gradient. 
SGD is an efficient algorithm that has the same 
sample complexity as regularized risk minimization.

## 14.1 Gradient descent

The gradient of a differentiable function $f(w)$ is denoted
as $\nabla f(w)$. It is a vector of partial derivatives
(one derivative per dimension). Gradient descent is an iterative algorithm:
We start with some value for $w$. Then for each subsequent iteration
we move by 

$$
w^{t + 1} = w^t - \eta \nabla f(w^t) 
$$

The gradient points in the direction of the greatest increase around $w$
for function $f$, we move in the opposite direction to 
decrease the value of the function (we wish to decrease the value of
the loss). After $T$ iterations, the algorithm outputs 
the averaged vector $w = \frac{1}{T} \sum_{t=1}^{T} w^t$
