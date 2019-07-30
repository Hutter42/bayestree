# bayestree
BayesTree

Given i.i.d. data from an unknown distribution, 
BayesTree is a non-parametric density estimator.

It is an adaptive way to estimate
the probability density by recursively subdividing the domain to
an appropriate data-dependent granularity. Bayes assigns
a data-independent prior probability to ``subdivide'', 
which leads to a prior over infinite(ly many) trees. 

This repository contains a simple inference algorithm for such a prior, 
for the data evidence, the predictive distribution, the effective model
dimension, moments, and other quantities. The code is efficient
for a single query in time, but for quering multiple points,
one should modify the algorithm and presort the data 
and only update one path in time O(log n),
rather than the whole tree in time O(n log n).
Alternatively one could vectorize the code for quering multiple points.
The code - as is - generates the data for all graphs in the paper
bayestreex.pdf and for the Excel sheets *.xls.

See paper bayestree.pdf for a more detailed explanation of the theory
behind BayesTree, the derivation of the algorithm, and 
convergence and asymptotic consistency proofs.
