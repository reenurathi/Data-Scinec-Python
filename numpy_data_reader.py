# import packages
import numpy as np
import os


def read_csv():
    # Using absolute path of file
    # os.path.dirname take the absolute path of any file and returns its directory path
    # os.path.realpath return the canonical path of the specified filename
    # __file__ gives the name of current file.
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = dir_path + "\\CSVDataForAssignment.csv"

    # np.loadtxt : Load data from a text file.
    # Parameters used:
    #   fname : [file, str, or pathlib.Path] File, filename, or generator to read.
    #   dtype : [dtype, optional] Data type of the resulting array.
    #   delimiter : [str, optional] The string used to separate values.
    #   skiprows : [int, optional] Skip the first skiprows lines; default: 0.
    #   usecols : [usecols : int or sequence, optional] Which columns to read, with 0 being the first.
    # Result:
    #   my_data : Content of file in form of N x N matrix.
    my_data = np.loadtxt(data_path, delimiter=",", dtype=object)

    # Saving first row as header.
    header = my_data[0] 

    # print my_data
    # print(my_data.shape)
    # return header, my_data[1:10]
    # Skipping first row as its is header
    return header, my_data[1:]

#FUNCTION FOR BASIC OPERATION ON ARREY

def Arrey_Operations():
    # Creating array object
    arr = np.array([[2, 4, 6, 8], [1, 3, 5, 7]])

    # Printing type of arr object
    print("Array is of type: ", type(arr))

    # Printing count of array dimensions (axes)
    print("No. of dimensions: ", arr.ndim)

    # Printing shape of array
    print("Shape of array: ", arr.shape)

    # Printing size (total number of elements) of array
    print("Size of array: ", arr.size)

    # Printing type of elements in array
    print("Array stores elements of type: ", arr.dtype)

#Function for crfeating an arrey from tuple and list and perfoeming function of numpy like random, full, zero
# or reshaping arrey

def createArr():
    # Python program to demonstrate array creation techniques

    # Creating array from list with type float
    a = np.array([[1, 2, 4], [5, 8, 7]], dtype='float')
    print ("Array created using passed list:\n", a)

    # Creating array from tuple
    b = np.array((1, 3, 2))
    print ("\nArray created using passed tuple:\n", b)

    # Creating a 3X4 array with all zeros
    c = np.zeros((3, 4))
    print ("\nAn array initialized with all zeros:\n", c)

    # Create a constant value array of complex type
    d = np.full((3, 3), 6, dtype='complex')
    print ("\nAn array initialized with all 6s. Array type is complex:\n", d)

    # Create an array with random values
    e = np.random.random((2, 2))
    print ("\nA random array:\n", e)

    # Create a sequence of integers
    # from 0 to 30 with steps of 5
    f = np.arange(0, 30, 5)
    print ("\nA sequential array with steps of 5:\n", f)

    # Create a sequence of 10 values in range 0 to 5
    g = np.linspace(0, 5, 10)
    print ("\nA sequential array with 10 values between 0 and 5:\n", g)

    # Reshaping 3X4 array to 2X2X3 array
    arr = np.array([[1, 2, 3, 4], [5, 2, 4, 2], [1, 2, 0, 1]])

    newarr = arr.reshape(2, 2, 3)

    print ("\nOriginal array:\n", arr)
    print ("Reshaped array:\n", newarr)

    # Flatten array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    flarr = arr.flatten()

    print ("\nOriginal array:\n", arr)
    print ("Fattened array:\n", flarr)

#indexing of array
def arrIndex():
    # An exemplar array
    arr = np.array([[-1, 2, 0, 4], [4, -0.5, 6, 0], [2.6, 0, 7, 8], [3, -7, 4, 2.0]])

    # Slicing array
    temp = arr[:2, ::2]
    print ("Array with first 2 rows and alternate columns(0 and 2):\n", temp)

    # Integer array indexing example
    temp = arr[[0, 1, 2, 3], [3, 2, 1, 0]]
    print ("\nElements at indices (0, 3), (1, 2), (2, 1), (3, 0):\n", temp)

    # boolean array indexing example
    cond = arr > 0  # cond is a boolean array
    temp = arr[cond]
    print ("\nElements greater than 0:\n", temp)

#arthmatic operations on single arreys
def calcArr():
    a = np.array([1, 2, 5, 3])

    # add 1 to every element
    print ("Adding 1 to every element:", a + 1)

    # subtract 3 from each element
    print ("Subtracting 3 from each element:", a - 3)

    # multiply each element by 10
    print ("Multiplying each element by 10:", a * 10)

    # square each element
    print ("Squaring each element:", a ** 2)

    # modify existing array
    a *= 2
    print ("Doubled each element of original array:", a)

    # transpose of array
    a = np.array([[1, 2, 3], [3, 4, 5], [9, 6, 0]])

    print ("\nOriginal array:\n", a)
    print ("Transpose of array:\n", a.T)

# using the urinary operator on arreys
def UrinaryOperatorArr():
    a = np.array([[1, 2],
                  [3, 4]])
    b = np.array([[4, 3], [2, 1]])

    # add arrays
    print ("Array sum:\n", a + b)

    # multiply arrays (element wise multiplication)
    print ("Array multiplication:\n", a * b)

    # matrix multiplication
    print ("Matrix multiplication:\n", a.dot(b))

#sorting of arrey
def SortArr():
    a = np.array([[1, 4, 2], [3, 4, 6], [0, -1, 5]])

    # sorted array
    print ("Array elements in sorted order:\n", np.sort(a, axis=None))

    # sort array row-wise
    print ("Row-wise sorted array:\n", np.sort(a, axis=1))

    # specify sort algorithm
    print ("Column wise sort by applying merge-sort:\n", np.sort(a, axis=0, kind='mergesort'))

    # Example to show sorting of structured array

    # set alias names for dtypes
    dtypes = [('name', 'S10'), ('grad_year', int), ('cgpa', float)]

    # Values to be put in array
    values = [('aaaaa', 2009, 8.5), ('bbbbb', 2008, 8.7), ('ccccc', 2008, 7.9), ('ddddd', 2009, 9.0)]

    # Creating array
    arr = np.array(values, dtype=dtypes)
    print ("\nArray sorted by names:\n", np.sort(arr, order='name'))

    print ("Array sorted by grauation year and then cgpa:\n", np.sort(arr, order=['grad_year', 'cgpa']))
