# main file
from numpy_data_reader import *
from matplotlib_plot_data import *
from scipy_usage import *

def main():
    header, data = read_csv()
    Arrey_Operations()
    createArr()
    calcArr()
    UrinaryOperatorArr()
    SortArr()
#ploting simple graph without using the data set
    line_graph_plot()
    bar_plot()
    histogram_plot()
    scatter_plot()
    curve_plot()
#ploting the graphgs with used the external dataset
    plot_data(header, data)
    plot_histogram(data)
    file_io()
    linear_algebra_operation()
    scipy_interpolation()
    scipy_optimization_fit()
    scipy_statistics()
    scipy_image_manipulation()


if __name__ == "__main__":
    main()
