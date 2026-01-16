import numpy as np


def load_muram_subfield(mfile):
    ''' Load a subfield from the SPIN4D project

    INPUT
    -----
    :mfile: extracted subfield from the SPIN4D project containing a single simulation variable.  Array shape is [128,1536,1536].  First for numbers are number of fields, ylen, zlen, and time.
    
    OUTPUT
    ------
    :dat: loaded data aray
    
    '''
    nx,ny,nz = 128,1536,1536
    tmp = np.fromfile(mfile,dtype=np.float32).reshape((nz,ny,nx))
    tmp = tmp.transpose(2,0,1)
    return(tmp)

def minmax(x):
    return(np.array([np.min(x),np.max(x)]))

def normalize_data(data,sregion=0.9):
    ''' Normalize input data
    
    Input
    -----
    :data: 2D array to be normalized
    :sregion: central area to consider to determine the normalization.  Defaults to 0.9.  sregion=1 means use everything
    
    Output
    ------
    :norm_data: Data normalized by minimum subtraction and dividing by the range.
    '''
    
    sz = data.shape
    yl = int(sz[0]/2-sz[0]/2*sregion)
    yu = int(sz[0]/2+sz[0]/2*sregion)
    xl = int(sz[1]/2-sz[1]/2*sregion)
    xu = int(sz[1]/2+sz[1]/2*sregion)
    mm = minmax(data[yl:yu,xl:xu])
    norm_data = (data-mm[0])/(mm[1]-mm[0])
    return (norm_data)


def get_props_from_ndat(ndat,threshold=0.37):
    ''' get granulation properties for data an array of intensities.
    
    INPUTS
    ndat - an array of normalized intensities I(x,y), size NX x NY
    threshold - intensity threshold.  pixels below the threshold will all be given a label of 0.

    RETURNS
    gprops - skimage regionprops object (can calculate areas, centroids, perimeters, etc, of each labeled region)
    labels - label array L(x,y), each pixel given the integer number of its region
    '''
    from skimage.segmentation import watershed # image processing library: segmentation algorithm
    from skimage.feature import peak_local_max # find local maxima
    from scipy import ndimage as ndi # multidimensional imageprocessing tools
    from skimage.measure import regionprops
    
    # find local maxima
    gcoords = peak_local_max (np.abs(ndat),threshold_abs=0.5, min_distance=8)
    
    # create a structure to use as a mask.  it will be False everywhere except at the local maxima pixels:
    mask = np.zeros(ndat.shape, dtype=bool)
    # set the maxima to True.  The transpose (.T) is required because some of the routines work in coordinates where 
    # (first index, second index) = (row, column) while others use the reverse
    mask[tuple(gcoords.T)] = True

    # name each of the local maxima by assigning them numbers
    markers, _ = ndi.label(mask)
    
    # create another mask to use with the watershed algorithm
    watershed_mask = (ndat>threshold)
    
        # perform the watershed algorithm: assign all lower intensity pixels connected to a local maxima the same label as the maxima:
    labels = watershed(-ndat, markers=markers,mask = watershed_mask)
    
        # use the regionprops function to generate properties all each labeled region (individual granules)
    gprops = regionprops(labels,intensity_image=ndat)
        
    return(gprops, labels)


# The following are pulled from https://stackoverflow.com/questions/25379752/how-can-i-extract-the-boundary-curve-of-an-image-region-in-scikit-image
# and are based on the code at https://github.com/theobdt/boundary_tracing/blob/master/bt.py

def moore_neighborhood(current, backtrack):  # y, x
    """Returns clockwise list of pixels from the moore neighborhood of current\
    pixel:
    The first element is the coordinates of the backtrack pixel.
    The following elements are the coordinates of the neighboring pixels in
    clockwise order.

    Parameters
    ----------
    current ([y, x]): Coordinates of the current pixel
    backtrack ([y, x]): Coordinates of the backtrack pixel

    Returns
    -------
    List of coordinates of the moore neighborood pixels, or 0 if the backtrack
    pixel is not a current pixel neighbor
    """

    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1],
                           [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def boundary_tracing(region):
    """Coordinates of the region's boundary. The region must not have isolated
    points (e.g., be simply connected).

    Parameters
    ----------
    region : obj
        Obtained with skimage.measure.regionprops()

    Returns
    -------
    boundary : 2D array
        List of coordinates of pixels in the boundary
        The first element is the most upper left pixel of the region.
        The following coordinates are in clockwise order (output is a directed graph).
    """

    # creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # initilization
    # starting point is the most upper left point
    idx_start = 0
    while True:  # asserting that the starting point is not isolated
        start = [y[idx_start], x[idx_start]]
        focus_start = binary[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
        if np.sum(focus_start) > 1:
            break
        idx_start += 1

    # Determining backtrack pixel for the first element
    if (binary[start[0] + 1, start[1]] == 0 and
            binary[start[0]+1, start[1]-1] == 0):
        backtrack_start = [start[0]+1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0

    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx-1]
        current = neighbors_current[idx]
        counter += 1

        if (np.all(current == start) and np.all(backtrack == backtrack_start)):
            break

    return np.array(boundary)