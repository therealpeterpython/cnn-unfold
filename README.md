# cnn-unfold
A lot of this explanation is taken from [here](https://stackoverflow.com/a/44039201).

This repository provides some python code which is able to unfold a convolutional kernel or the data matrix.
What do i mean with unfolding the kernel?

Let's say you have a 2d input `img` and 2d kernel `ker` and you want to calculate the convolution `img * ker`. Let's assume that `img` is of size `n×n` and `ker` is `k×k`.

So you unroll `ker` into a sparse matrix of size `(n-k+1)^2 × n^2`, and unroll `img` into a long vector `n^2 × 1`. You compute a multiplication of this sparse matrix with a vector and convert the resulting vector (which will have a size `(n-k+1)^2 × 1`) into a `n-k+1` square matrix.

I am pretty sure this is hard to understand just from reading. So here is an example for 2×2 kernel and 3×3 input.

$$
\left(\begin{array}{ccc}
x_1 & x_2 & x_3\\
x_4 & x_5 & x_6\\
x_7 & x_8 & x_9
\end{array}\right)
*
\left(\begin{array}{cc}
k_1 & k_2\\
k_3 & k_4
\end{array}\right)
$$

Here is the unfolded kernel matrix with the data vector:

$$
\left(\begin{array}{ccccccccc}
k_1 & k_2 & 0 & k_3 & k_4 & 0 & 0 & 0 & 0\\
0 & k_1 & k_2 & 0 & k_3 & k_4 & 0 & 0 & 0\\
0 & 0 & 0 & k_1 & k_2 & 0 & k_3 & k_4 & 0\\
0 & 0 & 0 & 0 & k_1 & k_2 & 0 & k_3 & k_4
\end{array}\right)
\cdot
\left(\begin{array}{c}
x_1\\
x_2\\
x_3\\
x_4 \\
x_5 \\
x_6 \\
x_7 \\
x_8 \\
x_9
\end{array}\right)
$$

which is equal to $\left(\begin{array}{c}
k_1 x_1 + k_2 x_2 + k_3 x_4 + k_4 x_5\\
k_1 x_2 + k_2 x_3 + k_3 x_5 + k_4 x_6\\
k_1 x_4 + k_2 x_5 + k_3 x_7 + k_4 x_8\\
k_1 x_5 + k_2 x_6 + k_3 x_8 + k_4 x_9
\end{array}\right)$.

And this is the same result (vectorized) you would have got by doing a sliding window of `ker` over `img`.

If we want to express the convolution as a matrix product but we want to unfold the data matrix and not the kernel then this is possible too.
You have to unroll your image to a matrix of size `(n-k+1)^2 x k^2` and the kernel to a `k^2 x 1` vector.

Let's take our example with the 2x2 kernel and the 3x3 data matrix from above.
$$
\left(\begin{array}{ccc}
x_1 & x_2 & x_3\\
x_4 & x_5 & x_6\\
x_7 & x_8 & x_9
\end{array}\right)
*
\left(\begin{array}{cc}
k_1 & k_2\\
k_3 & k_4
\end{array}\right)
$$

Here is the unfolded data matrix with the kernel vector:

$$
\left(\begin{array}{cccc}
x_1 & x_2 & x_4 & x_5\\
x_2 & x_3 & x_5 & x_6\\
x_4 & x_5 & x_7 & x_8\\
x_5 & x_6 & x_8 & x_9
\end{array}\right)
\cdot
\left(\begin{array}{c}
k_1\\
k_2\\
k_3\\
k_4
\end{array}\right)
$$


which is equal to $\left(\begin{array}{c}
x_1 k_1 + x_2 k_2 + x_4 k_3 + x_5 k_4\\
x_2 k_1 + x_3 k_2 + x_5 k_3 + x_6 k_4\\
x_4 k_1 + x_5 k_2 + x_7 k_3 + x_8 k_4\\
x_5 k_1 + x_6 k_2 + x_8 k_3 + x_9 k_4
\end{array}\right)$.

And this is the same result as above.




### Functions

`get_unfolded_kernel(ker, n)`:
Simply unfolds your kernel `ker` for the image size `n` and saves a template for faster calculation in the future. The method uses the saved templates if possible.

`generate_template_set(sizes=[(28,1,28), (48,1,48), (56,1,56)])`:
Generates and saves templates for the given image and kernel sizes.
You can use this in advance to generate templates for the sizes you need. The calculation is much faster when you have a template. The templates are index matrices for the unfolded kernels with a `-1` for static padding zeros.
You can call this method with a list of tuples as argument.
Each tuple consists of the image size, the minimal kernel size and the maximal kernel size. The method will then generate templates for all kernel sizes in between.

`unfold_data(x, k)`:
Unfolds a given data matrix `x` for the kernel size `k`.
This currently doesn't use templates.


Every kernel template gets saved in `./uf_kernel_tpls/` with the name `ufkernel_n_k.tpl` where `k` is the kernel and `n` is the image size.

If you use this code it would be great if you mention this repository.

