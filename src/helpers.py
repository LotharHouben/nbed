import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Parabola2D(x, c0, c1, c2,c3,c4,c5):
    """ Second-order polynomial function in 2D 

    Parameters:
    x:     two-dimensional array
    c0:    constant
    c1,c2: linear coefficients
    c3,c4: quadratic coefficients
    c5:    bilinear coefficient

    Returns:

    2D array with the two-dimensional parabola

    """
    return c0+c1*x[0]+c2*x[1]+c3*x[0]*x[0]+c4*x[1]*x[1]+c5*x[0]*x[1]

def ParabolaFit2D(zdata):
    """ Fit a second-order polynomial function to a 2D surface defined by zdata

    Parameters:
    zdata:     two-dimensional array with surface or intensity values
  

    Returns:

    2D array with the fitted two-dimensional parabola values

    """
    dim=zdata.shape # get dimensions
    # prepare support xdata
    limits = [0, 1., 0, 1.]  # [x1_min, x1_max, x2_min, x2_max]
    side_x = np.linspace(limits[0], limits[1], dim[0])
    side_y = np.linspace(limits[2], limits[3], dim[1])
    X1, X2 = np.meshgrid(side_y, side_x)
    size = X1.shape
    x1_1d = X1.reshape((1, np.prod(size)))
    x2_1d = X2.reshape((1, np.prod(size)))
    xdata = np.vstack((x1_1d, x2_1d))
    func=Parabola2D
    z=zdata.reshape(dim[0]*dim[1])
    popt, pcov = curve_fit(func, xdata, z)
    z_fit = func(xdata, *popt)
    Z_fit = z_fit.reshape(size)
    print(z.shape)
    return Z_fit

# 2D convolution. Source: https://medium.com/analytics-vidhya/2d-convolution-using-python-numpy-43442ff5f381
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        # print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break
    #print('before')
    return output

def bytscl(img,vmin=None,vmax=None):
    if vmin is None:
        vmin=np.min(img)
    if vmax is None:
        vmax=np.max(img)
    img8bit=np.copy(img)
    img8bit[img > vmax]=vmax
    img8bit[img < vmin]=vmin
    img8bit=255/(vmax-vmin)*(img8bit-vmin)
    return np.uint8(img8bit)    

def read_empad(fname):
    """
    Reads the EMPAD file at filename, returning a 4D numpy array .

    EMPAD files are 130x128 arrays, consisting of 128x128 arrays of data followed by
    two rows of metadata.  The metadata holds the scan position.
    The function determines the scan size by extracting the first and last frames' scan position.
    Then the data set is shaped.

    Arguments:
        fname     path to the EMPAD file

    Returns:
        data      datacube, excluding the metadata rows.
    """
    print("Reading EMPAD raw data file "+fname)
    rows = 130
    cols = 128
    filesize = os.path.getsize(fname)
    framesize = rows * cols * 4  # 4 bytes per pixel
    NFrames = filesize / framesize
    data_shape = (int(NFrames), rows, cols)
    with open(fname, "rb") as fid:
        data = np.fromfile(fid, np.float32).reshape(data_shape)[:, :rows, :]
    # Get the scan shape
    (dx,dy)=(1+data[-1,129:130,12]-data[0,129:130,12],1+data[-1,129:130,13]-data[0,129:130,13])
    data_shape=(int(dx.item()),int(dy.item()),rows,cols)  
    data=data.reshape(data_shape)
    return data[:,:,0:128,:]

