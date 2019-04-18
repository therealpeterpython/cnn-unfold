import numpy as np
import json
import os


"""
First written at 11.04.2019.

The main part of this script is the implementation of an unfolding method for kernel matrices which get's explained in
https://stackoverflow.com/a/44039201.

There is an implementation for unfolding the data matrix too. The method get's explained in
http://www.telesens.co/2018/04/09/initializing-weights-for-the-convolutional-and-fully-connected-layers/
and the code is from there and https://stackoverflow.com/a/49741532.
The method 'unfold_data(x, k)' was written by 'ankur6ue'.

Feel free to check out the github repo (https://github.com/therealpeterpython/cnn-unfold) too.
"""


"""
Loads an unfolded kernel template and calculates the unfolded kernel (faster) 
if the template was generated before, otherwise it generates it (slower).

:param ker: the kernel matrix which gets unfolded (kernel = k*k)
:param n: the size of the image (image = n*n)
"""
def get_unfolded_kernel(ker, n):
    _check_sizes(ker, n) # raise an error if our sizes are fishy
    k = ker.shape[0]
    ker_vec = ker.flatten()
    ker_uf = []

    file_path = "uf_kernel_tpls/ufkernel_{}_{}.tpl".format(n, k)
    if os.path.isfile(file_path):
        uf_kernel_tpl = _load_uf_kernel_tpl(k, n) # load the kernel template
    else:
        print("The kernel template wasn't in store so we have to generate it brand new!")
        uf_kernel_tpl = _generate_uf_kernel_template(k, n) # generate the kernel template
        print("The kernel get's saved for the next time!")
        _save_uf_kernel_tpl(uf_kernel_tpl, k, n) # and save it for the next time

    # use the kernel_tpl as index matrix (row wise)
    for i in range(uf_kernel_tpl.shape[0]):
        ker_uf.append(list(ker_vec[uf_kernel_tpl[i]]))

    ker_uf = np.asarray(ker_uf)
    ker_uf[uf_kernel_tpl  == -1] = 0    # place the static zero padding
    return np.asarray(ker_uf)


"""
Generates and saves kernel templates for the given image and kernel sizes

:param sizes: list of tuples with content (img_size, first_kernel_size, last_kernel_size) where all kernel 
              templates between first_kernel_size and last_kernel_size get's generated
:returns: none
"""
def generate_template_set(sizes=[(28,1,28), (48,1,48), (56,1,56)]):
    for size in sizes:
            for ker_size in range(size[1], size[2]):
                info = "Image size: {:>5} | Kernel size: {:>5}".format(size[0], ker_size)
                print(info)
                if not os.path.isfile("./uf_kernel_tpls/ufkernel_{}_{}.tpl".format(size[0], ker_size)): # if the template doesn't exist
                    ker = _generate_kernel_template(ker_size)
                    get_unfolded_kernel(ker, size[0]) # method returns the unfolded kernel and saves it


"""
Unfolds the data matrix x with respect to the kernel size k.
Written by 'ankur6ue' (more details in the description above).

:param x: the data matrix
:param k: the size of the kernel (kernel = k*k)
"""
def unfold_data(x, k):
    n, m = x.shape[0:2]
    xx = np.zeros(((n - k + 1) * (m - k + 1), k**2))
    row_num = 0
    def make_row(mat):
        return mat.flatten()

    for i in range(n - k+ 1):
        for j in range(m - k + 1):
            #collect block of m*m elements and convert to row
            xx[row_num,:] = make_row(x[i:i+k, j:j+k])
            row_num = row_num + 1

    return xx



# \/\/\/ Private functions for the internal use \/\/\/


"""
Unfolds a kernel. You can use this method by hand but then it will always 
generate(!) the unfolded kernel. The method get_unfolded_kernel unfolds a kernel 
too but checks first if there is a template to load.

:param ker: the kernel matrix
:param n: the size of the image (image = n*n)
"""
def _unfold_kernel(ker, n):
    _check_sizes(ker, n) # checks if the sizes are valid

    k = ker.shape[0]
    rows = []
    first_row = []

    # create the first row out of kernel elements and zeros
    for i in range(k):
        first_row.extend(np.append(ker[i], np.zeros(n-k)))
    first_row.extend(np.zeros(n**2-n*k)) # extend the first row with zeros

    # create the other rows by shifting (rolling) the first one
    for i in range((n-k+1)):
        for j in range((n-k+1)):
            rows.append(np.roll(first_row, i*n+j))

    ker_unfold = np.asarray(rows)

    return ker_unfold


"""
Just unfolds a kernel template to get a template for the unfolded kernel with 
respect to the image and kernel size

:param k: is the kernel size (kernel = k*k)
:param n: is the image size (image = n*n)
:returns: an index matrix with negative entries for zero padding(not the index zero)
"""
def _generate_uf_kernel_template(k, n):
    ker = _generate_kernel_template(k)
    uf_ker = _unfold_kernel(ker, n) - 1
    return uf_ker.astype(int)


"""
Generates a kernel template with the size k*k.
i.e. for k=2: return np.asarray([[1,2],[3,4]])

:param k: is the size of the kernel (kernel = k*k)
"""
def _generate_kernel_template(k):
    tpl = []
    for i in range(k):
        tpl.append(np.arange(k) + 1 +i*k)
    return np.asarray(tpl)


"""
Checks if the sizes are plausible (shouldn't fail).

:param ker: is the kernel
:param n: the size of the image
"""
def _check_sizes(ker, n, ker_orig_size=-1):
    if ker_orig_size == -1:  # ker is not an unfolded kernel
        if not ker.shape[0] == ker.shape[1]:
            raise ValueError("the kernel has to be a square matrix!")
        if ker.shape[0] > n:
            raise ValueError("the kernel has to be smaller then the image!")
    else:  # ker is an unfolded kernel
        # size has to be (n - k + 1)^2 × n^2
        if not (ker.shape[0] == (n - ker_orig_size + 1) ** 2 and ker.shape[1] == n ** 2):
            raise ValueError("the unfolded kernel has the wrong size for this image and kernel size!")


"""
Converts an iterable of iterables (i.e. arrays) to a list of lists
i.e. for array(array([1,2]), array([3,4])): return [[1,2],[3,4]]

:param arr: iterable of iterables
"""
def _convert_to_lists(arr):
    out = []
    for a in arr:
        out.append(list(a))
    return out


"""
Used for the json module to serialize numpy.int64
actual unused because i convert everything in _save_uf_kernel_tpl to int (via 'default = int')

:param o: value to convert
"""
def _convert_to_int(o):
    if isinstance(o, np.int64):
        return int(o)
    else:
        return o


"""
Saves the unfolded kernel template in './uf_kernel_tpls/ufkernel_n_k.tpl'

:param k: kernel size
:param n: image size
"""
def _save_uf_kernel_tpl(ker, k, n):
    _check_sizes(ker, n, ker_orig_size=k)
    if not os.path.exists("./uf_kernel_tpls"):
        os.mkdir("./uf_kernel_tpls")

    file_path = "./uf_kernel_tpls/ufkernel_{}_{}.tpl".format(n, k)
    ker_json = json.dumps(_convert_to_lists(ker), default = int)
    with open(file_path, "w") as write_file:
        write_file.write(ker_json)


"""
Loads an unfolded kernel template from ./uf_kernel_tpls/ufkernel_n_k.tpl'

:param k: kernel size
:param n: image size
"""
def _load_uf_kernel_tpl(k, n):
    file_path = "./uf_kernel_tpls/ufkernel_{}_{}.tpl".format(n, k)
    with open(file_path, "r") as write_file:
        ker_tpl = json.load(write_file)
    return np.asarray(ker_tpl)
