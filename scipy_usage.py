from scipy import io as spio
from scipy.interpolate import interp1d
from scipy import misc
from scipy import stats
from scipy import linalg
from scipy import optimize
from scipy import ndimage
from matplotlib import pyplot as plt
import numpy as np


def file_io():
    #  np.ones return a new array of given shape and type, filled with ones.
    a = np.ones(3)
    #  Save a dictionary of names and arrays into a MATLAB-style .mat file using key as 'a'.
    spio.savemat('file.mat', {'a': a})
    #  Load MATLAB file value for key as 'a'.
    data = spio.loadmat('file.mat')['a']
    print data

    #  Read an image from a file as an array.
    #  This function is only available if Python Imaging Library (PIL) is installed.
    # misc.imread('acron.png')

    # Matplotlib also has a similar function
    im_read = plt.imread('C:\Users\pc\Desktop\DataScience_project-master\horse.png')
    print im_read


def linear_algebra_operation():
    arr = np.array([[1, 2], [3, 4]])

    #  Compute the determinant of a matrix
    #  The determinant of a square matrix is a value derived arithmetically from the coefficients of the matrix.
    linalg.det(arr)
    arr = np.array([[3, 2], [6, 4]])
    linalg.det(arr)

    #  The determinant of a matrix can be calculated if matrix is a square matrix.
    #  linalg.det(np.ones((3, 4)))

    #  Compute the inverse of a matrix.
    arr = np.array([[1, 2], [3, 4]])
    iarr = linalg.inv(arr)
    print iarr
    np.allclose(np.dot(arr, iarr), np.eye(2))

    #  computing the inverse of a singular matrix (its determinant is zero) will raise LinAlgError:
    #  arr = np.array([[3, 2], [6, 4]])
    #  linalg.inv(arr)

    #  calculating singular-value decomposition
    arr = np.arange(9).reshape((3, 3)) + np.diag([1, 0, 1])
    uarr, spec, vharr = linalg.svd(arr)
    #  result as an array spectrum
    print spec

    #  now, the original matrix can be re-composed
    #  by matrix multiplication of the outputs of singular-value decomposition with np.dot (dor product of matrix)
    sarr = np.diag(spec)  # Extracting diagonal or construct a diagonal array
    svd_mat = uarr.dot(sarr).dot(vharr)
    print np.allclose(svd_mat, arr)  # Returns True if two arrays are element-wise equal within a tolerance


def scipy_interpolation():
    #  generating data
    np.random.seed(0)
    #  Returning evenly spaced numbers for specified interval
    measured_time = np.linspace(0, 1, 10)
    #  preparing deviation or noise in data
    noise = (np.random.random(10) * 2 - 1) * 1e-1
    #  imagining experimental data to a sine function
    measures = np.sin(2 * np.pi * measured_time) + noise

    #  building a linear interpolation function for 1-d function with sne measures
    linear_interp = interp1d(measured_time, measures)  # Interpolate a 1-D function
    interpolation_time = np.linspace(0, 1, 50)
    linear_results = linear_interp(interpolation_time)
    cubic_interp = interp1d(measured_time, measures, kind='cubic')
    cubic_results = cubic_interp(interpolation_time)

    # Plot the data and the interpolation
    plt.figure(figsize=(6, 4))
    plt.plot(measured_time, measures, 'o', ms=6, label='measures')
    plt.plot(interpolation_time, linear_results, label='linear interp')
    plt.plot(interpolation_time, cubic_results, label='cubic interp')
    plt.legend()
    plt.show()


def scipy_optimization_fit():
    # Seed the random number generator for reproducibility
    np.random.seed(0)
    #  Returning evenly spaced numbers for specified interval
    x_data = np.linspace(-5, 5, num=50)
    #  imagining experimental data to a sine function
    y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)

    # Plotting curve for prepared data
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data)
    plt.show()

    #  fit a simple sine function to the data
    #  Defining inner function
    def test_func(x, a, b):
        return a * np.sin(b * x)

    #  Use non-linear least squares to fit a function, f, to data.
    params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
    print(params)

    #  plot the resulting curve on the data
    plt.figure(figsize=(6, 4))
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, test_func(x_data, params[0], params[1]), label='Fitted function')
    plt.legend(loc='best')
    plt.show()


def scipy_statistics():
    # Sample from a normal distribution using numpy's random number generator
    samples = np.random.normal(size=10000)

    # Compute a histogram of the sample
    # using evenly spaced numbers for specified interval
    bins = np.linspace(-5, 5, 30)

    #  VisibleDeprecationWarning: Passing `normed=True` on non-uniform bins has always been broken,
    #  and computes neither the probability density function nor the probability mass function.
    #  The result is only correct if the bins are uniform, when density=True will produce the same result anyway.
    #  The argument will be removed in a future version of numpy.

    #  histogram, bins = np.histogram(samples, bins=bins, normed=True)  # Compute the histogram of a set of data.

    # Although Without normed=True, PDF gives flat line for linear sample data
    histogram, bins = np.histogram(samples, bins=bins)  # Compute the histogram of a set of data.
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Compute the Probability Density Function on the bin centers from scipy distribution object
    pdf = stats.norm.pdf(bin_centers)  # PDF over normal continuous random variable.

    plt.figure(figsize=(6, 4))
    plt.plot(bin_centers, histogram, label="Histogram of samples")
    plt.plot(bin_centers, pdf, label="PDF")
    plt.legend()
    plt.show()


def scipy_image_manipulation():
    #  Geometrical transformations of an images.

    #  Get a 1024 x 768, color image of a raccoon face. Image used is defalut library image.
    #  raccoon-procyon-lotor.jpg at http://www.public-domain-image.com
    face = misc.face(gray=True)

    # Applying a various transformations

    #  The array is shifted using spline interpolation of the requested order.
    #  Points outside the boundaries of the input are filled according to the given mode.
    shifted_face = ndimage.shift(face, (50, 50))
    shifted_face2 = ndimage.shift(face, (50, 50), mode='nearest')

    #  The array is rotated in the plane defined by the two axes
    #  given by the axes parameter using spline interpolation of the requested order.
    rotated_face = ndimage.rotate(face, 30)

    cropped_face = face[50:-50, 50:-50]  # cropping ndarray
    zoomed_face = ndimage.zoom(face, 2)  # The array is zoomed using spline interpolation of the requested order.

    print zoomed_face.shape

    #  Createing a new figure.
    plt.figure(figsize=(15, 3))
    #  Adding subplot in current figure.
    plt.subplot(151)
    #  Displaying an image as data on a 2D regular raster in gray color scheme
    plt.imshow(shifted_face, cmap=plt.cm.gray)
    #  Turning off the axis lines and labels.
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(shifted_face2, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(153)
    plt.imshow(rotated_face, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(154)
    plt.imshow(cropped_face, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(155)
    plt.imshow(zoomed_face, cmap=plt.cm.gray)
    plt.axis('off')
    #  adjusting various above defined sub-plots in single frame.
    plt.subplots_adjust(wspace=.05, left=.01, bottom=.01, right=.99, top=.99)

    plt.show()