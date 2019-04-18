# cnn-unfold
A lot of this explanation is taken from [here](https://stackoverflow.com/a/44039201).

This repository provides python code which is able to unfold a convolutional kernel or the data matrix.  
What do i mean with unfolding the kernel?

Let's say you have a 2d input `img` and 2d kernel `ker` and you want to calculate the convolution `img * ker`. Let's assume that `img` is of size `n×n` and `ker` is `k×k`.

So you unroll `ker` into a sparse matrix of size `(n-k+1)^2 × n^2`, and unroll `img` into a long vector `n^2 × 1`. You compute a multiplication of this sparse matrix with a vector and convert the resulting vector (which will have a size `(n-k+1)^2 × 1`) into a `n-k+1` square matrix.

I am pretty sure this is hard to understand just from reading. So here is an example for 2×2 kernel and 3×3 input.

![1]

Here is the unfolded kernel matrix with the data vector:

![2]

which is equal to ![3].

And this is the same result (vectorized) you would have got by doing a sliding window of `ker` over `img`.

If we want to express the convolution as a matrix product but we want to unfold the data matrix and not the kernel then this is possible too.
You have to unroll your image to a matrix of size `(n-k+1)^2 x k^2` and the kernel to a `k^2 x 1` vector.

Let's take our example with the 2x2 kernel and the 3x3 data matrix from above.

![1]

Here is the unfolded data matrix with the kernel vector:

![4]


which is equal to ![5].

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


[1]: https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbg_white%20%5Clarge%20%24%24%5Cleft%28%5Cbegin%7Barray%7D%7Bccc%7D%20x_1%26x_2%26x_3%5C%5C%20x_4%26x_5%26x_6%5C%5C%20x_7%26x_8%26x_9%20%5Cend%7Barray%7D%5Cright%29%20*%20%5Cleft%28%5Cbegin%7Barray%7D%7Bcc%7D%20k_1%26k_2%5C%5C%20k_3%26k_4%20%5Cend%7Barray%7D%5Cright%29%24%24

[2]: https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbg_white%20%24%24%5Cleft%28%5Cbegin%7Barray%7D%7Bccccccccc%7D%20k_1%26k_2%260%26k_3%26k_4%260%260%260%260%5C%5C%200%26k_1%26k_2%260%26k_3%26k_4%260%260%260%5C%5C%200%260%260%26k_1%26k_2%260%26k_3%26k_4%260%5C%5C%200%260%260%260%26k_1%26k_2%260%26k_3%26k_4%20%5Cend%7Barray%7D%5Cright%29%20%5Ccdot%20%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20x_1%5C%5C%20x_2%5C%5C%20x_3%5C%5C%20x_4%5C%5C%20x_5%5C%5C%20x_6%5C%5C%20x_7%5C%5C%20x_8%5C%5C%20x_9%20%5Cend%7Barray%7D%5Cright%29%24%24

[3]: https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cbg_white%20%5Clarge%20%24%24%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20k_1x_1&plus;k_2x_2&plus;k_3x_4&plus;k_4x_5%5C%5C%20k_1x_2&plus;k_2x_3&plus;k_3x_5&plus;k_4x_6%5C%5C%20k_1x_4&plus;k_2x_5&plus;k_3x_7&plus;k_4x_8%5C%5C%20k_1x_5&plus;k_2x_6&plus;k_3x_8&plus;k_4x_9%20%5Cend%7Barray%7D%5Cright%29%24%24

[4]: https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbg_white%20%5Clarge%20%24%24%5Cleft%28%5Cbegin%7Barray%7D%7Bcccc%7D%20x_1%26x_2%26x_4%26x_5%5C%5C%20x_2%26x_3%26x_5%26x_6%5C%5C%20x_4%26x_5%26x_7%26x_8%5C%5C%20x_5%26x_6%26x_8%26x_9%20%5Cend%7Barray%7D%5Cright%29%20%5Ccdot%20%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20k_1%5C%5C%20k_2%5C%5C%20k_3%5C%5C%20k_4%20%5Cend%7Barray%7D%5Cright%29%24%24

[5]: https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cdpi%7B100%7D%20%5Cbg_white%20%5Clarge%20%24%24%5Cleft%28%5Cbegin%7Barray%7D%7Bc%7D%20x_1k_1&plus;x_2k_2&plus;x_4k_3&plus;x_5k_4%5C%5C%20x_2k_1&plus;x_3k_2&plus;x_5k_3&plus;x_6k_4%5C%5C%20x_4k_1&plus;x_5k_2&plus;x_7k_3&plus;x_8k_4%5C%5C%20x_5k_1&plus;x_6k_2&plus;x_8k_3&plus;x_9k_4%20%5Cend%7Barray%7D%5Cright%29%24%24
