# import packages
import numpy as np
import matplotlib.pyplot as plt
import collections

np.set_printoptions(threshold=np.nan)

# plt(x, y)         # plot x and y using default line style and color
# plt(x, y, 'bo')   # plot x and y using blue circle markers
# plt(y)            # plot y using x as index array 0..N-1
# plt(y, 'r+')      # plot y using x as index array 0..N-1, but with red plusses
#
# https://matplotlib.org/2.2.3/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot


def plot_data(header, data):
    # Plot the data

    # Filtering data for country == "India"
    filter_data = data["India" == data[:, 2]]  

    # plot title with name "sales-rank graph"
    plt.title('sales-rank graph')

    # Labeling x axis with name "rank"
    plt.xlabel('rank')

    # Labeling y axis with name "sales"
    plt.ylabel('sales')

    # Plotting graph for column 0 as x axis and column 4 as y axis.
    plt.plot(filter_data[:, 0], map(float, filter_data[:, 4]))

    # Showing plotted graph.
    plt.show()

    # plot title with name "Profit-rank graph"
    plt.title('Profit-rank graph')

    # Labeling x axis with name "rank"
    plt.xlabel('rank')  

    # Labeling y axis with name "profit"
    plt.ylabel('profit')  

    # Plotting graph for column 0 as x axis and column 5 as y axis.
    plt.plot(filter_data[:, 0], map(float, filter_data[:, 5]))  

    # Showing plotted graph.
    plt.show()  

    # plot title with name "asset-rank graph"
    plt.title('asset-rank graph')

    # Labeling x axis with name "rank"
    plt.xlabel('rank')  

    # Labeling y axis with name "asset"
    plt.ylabel('asset')  

    # Plotting graph for column 0 as x axis and column 6 as y axis.
    plt.plot(filter_data[:, 0], map(float, filter_data[:, 6]))  

    # Showing plotted graph.
    plt.show()  

    # plot title with name "marketValue-rank graph"
    plt.title('marketValue-rank graph')

    # Labeling x axis with name "rank"
    plt.xlabel('rank')  

    # Labeling y axis with name "MarketValue"
    plt.ylabel('marketValue')  

    # Plotting graph for column 0 as x axis and column 7 as y axis.
    plt.plot(filter_data[:, 0], map(float, filter_data[:, 7]))  

    # Showing plotted graph.
    plt.show()  

    # https: // docs.python.org / 2 / library / collections.html  # collections.Counter
    counter = collections.Counter(data[:, 3])

    # plot title with name "Categories-count graph"
    plt.title('Categories-count graph')

    # Labeling x axis with name "company count"
    plt.xlabel('No. of companies')  

    # Labeling y axis with name "Categories"
    plt.ylabel('Categories')

    # Plotting bar graph for column 0 as x axis and column 5 as y axis.
    # plt.bar(counter.keys(), counter.values())  
    plt.scatter(counter.values(), counter.keys())

    # Showing plotted graph.
    plt.show()  


# http://docs.astropy.org/en/stable/visualization/histogram.html
# https://matplotlib.org/2.2.3/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist
# https://matplotlib.org/2.2.3/gallery/statistics/histogram_features.html
def plot_histogram(data):
    filter_data = data
    # filter_data = data["India" == data[:, 2]]  # filter the data for country == "India"
    # Collecting the profit column data.
    # Do do so, first need to get the transpose of matrix.
    # since all the data will be in string format so applying float to all values of transposed data using map function.
    # profit_data = map(float, filter_data[:, 5])  # Collect the profit data
    profit_data = map(lambda x: float(0) if x == "NA" else float(x), filter_data[:, 5])

    # setting the ranges and no. of intervals
    rnge = (-20, 20)
    bins = 70

    # plotting a histogram
    plt.hist(profit_data, bins, rnge, color='green', histtype='bar', rwidth=.7)

    # Labeling x axis with name "profit"
    plt.xlabel('profit')

    # Labeling y axis with name "No of companies"
    plt.ylabel('No of companies')

    # plot title with name "Histogram of profit against company count"
    plt.title('Histogram of profit against company count')

    # Showing plotted graph.
    # plt.hist(profit_data)
    # plt.axis([0, 2000, 0, 200])
    # plt.grid(True)
    plt.show()


#BASIC GRAPH USING MATPLOTLIB with out the data set used
def line_graph_plot():
    # x axis values
    x = [1, 4, 2]

    # corresponding y axis values
    y = [3, 6, 1]

    # plotting the points as one line graph
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x-axis')

    # naming the y axis
    plt.ylabel('y-axis')

    # giving a title to graph
    plt.title('Sample graph')

    # function to show the plot
    plt.show()
    # line 2 points
    x2 = [1, 2, 3]
    y2 = [4, 1, 3]

    # plotting the line 2 points
    plt.plot(x2, y2, label="line 2")

    # naming the x axis
    plt.xlabel('x-axis')

    # naming the y axis
    plt.ylabel('y-axis')

    # giving a title to graph
    plt.title('Two lines on graph')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

#Ploting the bar graph

def bar_plot():
    # x-coordinates of left sides of bars
    left = [1, 2, 3, 4, 5]

    # heights of bars
    height = [15, 24, 36, 40, 25]

    # labels for bars
    tick_label = ['A', 'B', 'C', 'D', 'E']

    # plotting a bar chart
    plt.bar(left, height, tick_label=tick_label, width=0.8, color=['red', 'blue'])

    # naming the x-axis
    plt.xlabel('x-axis')

    # naming the y-axis
    plt.ylabel('y-axis')

    # plot title
    plt.title('Bar chart')

    # function to show the plot
    plt.show()

#ploting the histogram
def histogram_plot():

    # frequencies
    ages = [2, 5, 70, 40, 30, 45, 50, 45, 43, 40, 44, 60, 7, 13, 57, 18, 90, 77, 32, 21, 20, 40]

    # setting the ranges and no. of intervals
    range = (0, 50)
    bins = 10

    # plotting a histogram
    plt.hist(ages, bins, range, color='blue', histtype='bar', rwidth=0.8)

    # x-axis label
    plt.xlabel('age')

    # frequency label
    plt.ylabel('No. of people')

    # plot title
    plt.title('Histogram')

    # function to show the plot
    plt.show()

#ploting the scatter plot
def scatter_plot():

    # x-axis values
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # y-axis values
    y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]

    # plotting points as a scatter plot
    plt.scatter(x, y, label="stars", color="red", marker="*", s=30)

    # x-axis label
    plt.xlabel('x-axis')

    # frequency label
    plt.ylabel('y-axis')

    # plot title
    plt.title('scatter plot')

    # showing legend
    plt.legend()

    # function to show the plot
    plt.show()

#ploting the piechart():

def pie_plot():
    # defining labels
    vehicles = ['taxi', 'bike', 'cars', 'train']

    # portion covered by each label
    slices = [3, 7, 8, 6]

    # color for each label
    colors = ['r', 'y', 'g', 'b']

    # plotting the pie chart
    plt.pie(slices, labels=activities, colors=colors,
            startangle=90, shadow=True, explode=(0, 0, 0.1, 0),
            radius=1.2, autopct='%1.1f%%')

    # plotting legend
    plt.legend()

    # showing the plot
    plt.show()


#ploting the curves
def curve_plot():

    # setting the x - coordinates
    x = np.arange(0, 2 * (np.pi), 0.1)

    # setting the corresponding y - coordinates
    y = np.sin(x)

    # potting the points
    plt.plot(x, y)

    # function to show the plot
    plt.show()

