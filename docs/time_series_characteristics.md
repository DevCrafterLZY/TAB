**Pseudocode for five characteristics used in TAB.**

**Statistics of both multivariate and univariate datasets**.

 

## Trend:

The trend of a time series refers to the long-term changes or patterns that occur over time. Intuitively, it represents the general direction in which the data is moving. Referring to the explained variance, Trend Strength can be defined as in Algorithm 1. Seasonal and Trend decomposition using Loess (STL), which is a highly versatile and robust method for time series decomposition.

<div align="center">
    <img src="trend.png" alt="trend"  width="100%" />
</div>





## Seasonality:

Seasonality refers to the phenomenon where changes in a time series repeat at specific intervals. Algorithm 2 details the calculation process.

<div align="center">
    <img src="seasonal.png" alt="seasonal"  width="100%" />
</div>





## Stationarity:

Stationarity refers to the mean of any observation in a time series $X=\langle x_1,x_2,...,x_n\rangle$ is constant, and the variance is finite for all observations. Also, the covariance $\mathit{cov}(x_i,x_j)$ between any two observations $x_i$ and $x_j$ depends only on their distance $|j-i|$, i.e., $\forall i+r\leq n,j+r\leq n$ $(\mathit{cov}(x_i,x_j)=\mathit{cov}(x_{i+r},x_{j+r})).$ Strictly stationary time series are rare in practice. Therefore, weak stationarity conditions are commonly applied. In our paper, we also exclusively focus on weak stationarity.

<div align="center">
    <img src="stationarity.png" alt="stationarity"  width="100%" />
</div>





## Shifting:

Shifting refers to the phenomenon where the probability distribution of time series changes over time. This behavior can stem from structural changes within the system, external influences, or the occurrence of random events. As the value approaches 1, the degree of shifting becomes more severe. Algorithm 4 details the calculation process.

<div align="center">
    <img src="shifting.png" alt="shifting"  width="100%" />
</div>





## Transition:

Transition refers to the trace of the covariance of transition
matrix between symbols in a 3-letter alphabet. It captures the regular and identifiable fixed features present in a time series, such as the clear manifestation of trends, periodicity, or the simultaneous presence of both seasonality and trend. Algorithm 5 details the calculation process. 

<div align="center">
    <img src="transition.png" alt="transition"  width="100%" />
</div>




<div align="center">
    <img src="multivariate-datasets.png" alt="transition"  width="100%" />
</div>



<div align="center">
    <img src="univariate-datasets.png" alt="transition"  width="100%" />
</div>
