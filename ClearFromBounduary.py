import random
import numpy as np
def ClearFromBoundary(dim,margin,Nemitters):
# [indxy,indxy_cfb] = ClearFromBoundary(dim,margin,Nemitters)
# function picks indices that are margin pixels apart from the boundary to 
# prevent patch truncation in the resulting dataset.
#
# Inputs
# dim           -   the dimensions of the image
# margin        -   the desired margin pixels
# Nemitters     -   number of emitters to sample
#
#
# Written by Elias Nehme, 25/08/2017

    # dimensions of the image
    M = dim[0]
    N = dim[1]
    
    x = np.arange(0, M, 1)
    y = np.arange(0, N, 1)
    rows, cols = np.meshgrid(x, y, sparse=False)
    
    # valid rows and columns
    rows_cb = rows[(rows>=margin-1) & (cols>=margin-1) & (rows<=(M-margin-1)) & (cols<=(N-margin-1))]
    cols_cb = cols[(rows>=margin-1) & (cols>=margin-1) & (rows<=(M-margin-1)) & (cols<=(N-margin-1))]
 
    # chose "random" x-y locations from the valid set
    indx = random.sample(list(rows_cb),Nemitters)
    indy = random.sample(list(cols_cb), Nemitters)
    return (indx, indy)

