# UnitBlockAngular.jl

Specialized linear algebra for unit block-angular matrices.
It is primarily intended for use within [Tulip.jl](https://github.com/ds4dm/Tulip.jl)

## Overview

Uni block-block angular matrices have the form

$$
A =
\begin{bmatrix}
    B_{0} & B_{1} & B_{2} & \dots & B_{R}\\
    0 & e^{T} \\
    0 & & e^{T}\\
    \vdots & & & \ddots\\
    0 & & & & e^{T}
\end{bmatrix}
$$

This package exports the `UnitBlobkAngularMatrix` type, and extends the 5-arg `mul!`.


## Matrix-vector multiplication

$y = A \times x$ writes

$$
\large
\begin{bmatrix}
    y_{0}\\ y_{1}\\ y_{2} \\ \vdots\\ y_{R}
\end{bmatrix}
=
\begin{bmatrix}
    B_{0} & B_{1} & B_{2} & \dots & B_{R}\\
    0 & e^{T} \\
    0 & & e^{T}\\
    \vdots & & & \ddots\\
    0 & & & & e^{T}
\end{bmatrix}
\begin{bmatrix}
    x_{0}\\ x_{1}\\ x_{2} \\ \vdots\\ x_{R}
\end{bmatrix}
=
\begin{bmatrix}
    \displaystyle \sum_{i=0}^{R}B_{i}x_{i}\\
    e^{T}x_{1}\\
    e^{T}x_{2}\\
    \vdots\\
    e^{T}x_{R}
\end{bmatrix}
=
\begin{bmatrix}
    B_{0} x_{0} + B x_{-0}\\
    e^{T}x_{1}\\
    e^{T}x_{2}\\
    \vdots\\
    e^{T}x_{R}
\end{bmatrix}
$$

$x = A^{T} \times y$ writes

$$
\Large
\begin{bmatrix}
    x_{0}\\ x_{1}\\ x_{2} \\ \vdots\\ x_{R}
\end{bmatrix}
=
\begin{bmatrix}
    B_{0}^{T} & 0 &0 & \dots & 0\\
    B_{1}^{T} & e\\
    B_{2}^{T} &&e\\
    \vdots &&& \ddots\\
    B_{R}^{T} & &&&e
\end{bmatrix}
\begin{bmatrix}
    y_{0}\\ y_{1}\\ y_{2} \\ \vdots\\ y_{R}
\end{bmatrix}
=
\begin{bmatrix}
    B_{0}^{T}y_{0}\\
    B_{1}^{T}y_{0} + y_{1} e\\
    B_{2}^{T}y_{0} + y_{2} e\\
    \vdots\\
    B_{R}^{T}y_{0} + y_{R} e\\
\end{bmatrix}
=
\begin{bmatrix}
B_{0}^{T}y_{0}\\
B^{T} y_{-0}
\end{bmatrix}
+
\begin{bmatrix}
    0\\
    y_{1} e\\
    y_{2} e\\
    \vdots \\
    y_{R} e\\
\end{bmatrix}
$$

## Factorization

The normal equations write

$$
A \times \Theta \times A^{T} =
\begin{bmatrix}
    \sum_{i=0}^{R} B_{i} \Theta_{i} B_{i}^{T} & B_{1} \theta_{1} & B_{2}\theta_{2} & \dots & B_{R} \theta_{r}\\
    \theta_{1}^{T} B_{1}^{T} & e^{T}\theta_{1} \\
    \theta_{2}^{T} B_{2}^{T} & & e^{T}\theta_{2}\\
    \vdots & & & \ddots\\
    \theta_{R}^{T} B_{R}^{T}  & & & & e^{T}\theta_{R}
\end{bmatrix}
$$