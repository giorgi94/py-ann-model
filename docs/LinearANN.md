
Suppose we have
\[
    x = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \quad
    y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_m \end{bmatrix}
\]
and we have some activation function $\sigma$. We need to find $w$ and $b$ matrices
\[
    w = \begin{bmatrix}
        w_{11} & w_{12} & \cdots & w_{1n} \\
        w_{11} & w_{12} & \cdots & w_{1n} \\
        \vdots & \vdots & \ddots & \vdots \\
        w_{m1} & w_{m2} & \cdots & w_{mn} \\
    \end{bmatrix} \quad
    b = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_m \end{bmatrix}
\]
such that
\[
    \sigma(z) = \sigma(wx + b) = \hat{y} \approx y
\]
and we are not interested in exact solution. Therefore, we need some method to measure an error of approximation. For simplicity let's following funtion
\[
    C(y,\hat{y}) = \frac{1}{2}\sum_{j=1}^{m} (\hat{y}_{j} - y_{j})^2
\]
since $C(y,\hat{y}) = 0 \Rightarrow y=\hat{y}$.

\[
    \frac{\partial C}{\partial w_{jk}} =
        \frac{\partial z_{j}}{\partial w_{jk}}
        \frac{\partial \hat{y}_{j}}{\partial z_{j}}
        \frac{\partial C}{\partial \hat{y}_{j}}
        =
        x_{k} \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j})
\]

\[
    \frac{\partial C}{\partial b_{j}} =
        \frac{\partial z_{j}}{\partial b_{j}}
        \frac{\partial b_{j}}{\partial z_{j}}
        \frac{\partial C}{\partial \hat{y}_{j}}
        =
        \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j})
\]

\[
    \frac{\partial C}{\partial x_{k}} =
    \sum_{j=1}^{m}
        \frac{\partial z_{j}}{\partial x_{k}}
        \frac{\partial \hat{y}_{j}}{\partial z_{j}}
        \frac{\partial C}{\partial \hat{y}_{j}} =
    \sum_{j=1}^{m}
        w_{jk}
        \frac{\partial \hat{y}_{j}}{\partial z_{j}}
        \frac{\partial C}{\partial \hat{y}_{j}} =
    \sum_{j=1}^{m}
        w_{jk} \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j})
\]
the last one is when we want to correct input. Let's write it in a form of a matrix

\[
    \frac{\partial C}{\partial w} =
        \left[ \frac{\partial C}{\partial w_{jk}} \right]_{m\times n} =
        \left[ x_{k} \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j}) \right]_{m\times n} = \sigma'(z) (\hat{y} - y) x^T
\]

\[
    \frac{\partial C}{\partial b} =
        \left[ \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j}) \right]_{m\times 1}
        = \sigma'(z) (\hat{y} - y)
\]

\[
    \frac{\partial C}{\partial x} =
        \left[
            \sum_{j=1}^{m}
            w_{jk} \sigma^{\prime}(z_{j}) (\hat{y}_{j} - y_{j})
        \right]_{n \times 1}
        = w^T \sigma'(z) (\hat{y} - y)
\]


where

\[
    \sigma'(z) = \begin{bmatrix}
        \sigma'(z_1) & 0 & \cdots  & 0 \\
        0 & \sigma'(z_2) & \cdots  & 0 \\
        \vdots & \vdots  &  \ddots & 0 \\
        0 & 0 & \cdots & \sigma'(z_{m}) \\
    \end{bmatrix}_{m\times m}
\]







<div style="height: 500px"></div>