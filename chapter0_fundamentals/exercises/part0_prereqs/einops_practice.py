#%%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"

#%% 
if MAIN:
    arr = np.load(section_dir / "numbers.npy")
    print(arr[0].shape)
    display_array_as_img(arr[0])  # plotting the first image in the batch
#%%
# trying out some exercises from einops.rocks 

# why einops? -> better readability, understanding of tensor manipulation!

if MAIN:
    # swapping height and width 
    # can use short versions too - c h w
    arr_swapped = einops.rearrange(arr[0], "color height width -> color width height")
    #display_array_as_img(arr_swapped)

    # composition of axes
    arr_composed = einops.rearrange(arr, "b c h w -> c (b h) w")
    #display_array_as_img(arr_composed)

    # decomposition of axes 
    # reshaping b, b1=2 -> 2 rows height, b2=arr.shape[0]/2 = 3 -> 3 cols width
    arr_decomposed = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
    # display_array_as_img(arr_decomposed)

    # order of axes
    # (w b) vs (b w)
    # (w b) interleaves all the images, new_index = w_index * b + b_index
    # column 0 of image 0, column 0 of image 1, and so on 
    # (b w) does the images side by side, new_index = b_index * w + w_index
    # normal side by side setup 
    arr_order = einops.rearrange(arr, "b h w c -> h (w b) c")
    #display_array_as_img(arr_order)

    # einops.reduce 
    # makes a blurred average image
    arr_reduce = einops.reduce(arr.astype(float), "b c h w -> c h w", "mean")
    #display_array_as_img(arr_reduce)
    # can use mean, min, max, sum, prod
    # useful for pooling

    # repeating 
    arr_repeat = einops.repeat(arr[0], "c h w -> c h (repeat w)", repeat=3)
    display_array_as_img(arr_repeat)
# %%
if MAIN:
    # basically multiplying the batch size by the width -> displaying in a row 
    arr_stacked = einops.rearrange(arr, "b c h w -> c h (b w)")
    print(arr_stacked.shape)
    display_array_as_img(arr_stacked)  # plotting all images, stacked in a row
# %%
