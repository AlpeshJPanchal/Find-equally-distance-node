import os
import random
from sklearn.metrics import pairwise_distances
import numpy
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist



def read_csv_to_float_list(file_path):
    data_list = []
    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            float_row = [float(cell) for cell in row]  # Convert each cell to float
            data_list.append(float_row)
    return data_list
def WriteCsv(data,filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(data)

def EqualityNodedata(data,NumberOfNode):
    starting_point_ind = random.SystemRandom().randint(1, len(data))

    points_from_data = numpy.array([data[starting_point_ind]])

    for i in range(1,NumberOfNode):
        pairwise_distances_to_data = pairwise_distances (data, Y=points_from_data, metric='euclidean', n_jobs=-1)

        pairwise_distances_to_data = numpy.array(pairwise_distances_to_data)

        min_distances_to_data = numpy.amin(pairwise_distances_to_data, axis=1)

        k = min_distances_to_data.argmax()


        points_from_data = numpy.append(points_from_data, [data[k]], axis=0)
        # Visualization(data,points_from_data)

    # Assign each point to the closest centroid.
    assignments = np.argmin(cdist(data, points_from_data), axis=1)
    print(assignments)
    # Recompute the centroids.

    for i in range(NumberOfNode):
        print(data[assignments == i])
        points_from_data[i] = np.mean(data[assignments == i], axis=0)
    # Assign each point to the closest centroid.

    #Visualization(data, points_from_data)
    assignments = np.argmin(cdist(points_from_data, data), axis=1)

    print(assignments)

    for i in range(NumberOfNode):
        points_from_data[i] = data[assignments[i]]
    #Visualization(data, points_from_data)

    return points_from_data
def Visualization(data,points_from_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d');

    # Accessing x, y, and z coordinates from the points array
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', label='Original Points')
    # # Accessing x, y, and z coordinates from the equally_distributed_points array
    ax.scatter(points_from_data[:, 0], points_from_data[:, 1], points_from_data[:, 2], c='red',
               label='Equally Distributed Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_ylabel('z')

    plt.title('Original Points vs Equally Distributed Points')
    plt.legend()
    plt.show()
    plt.close()
def main():
    N = 20
    file_path = "C:\\Users\\Alpesh\\Downloads\\nodetemp_16_27_25.csv"
    directory = os.path.dirname(file_path)
    output_file_path = os.path.join(directory, "output.csv")
    data = read_csv_to_float_list(file_path)
    data = numpy.array(data)
    EqualityNode=EqualityNodedata(data,N)
    WriteCsv(EqualityNode,output_file_path)
if __name__=="__main__":
    main()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d');
    #
    # # Accessing x, y, and z coordinates from the points array
    # ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', label='Original Points')
    #
    # # Accessing x, y, and z coordinates from the equally_distributed_points array
    # ax.scatter(points_from_data[:, 0], points_from_data[:, 1], points_from_data[:, 2], c='red',
    #            label='Equally Distributed Points')
    #
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_ylabel('z')
    #
    # plt.title('Original Points vs Equally Distributed Points')
    # plt.legend()
    # plt.show()
    # plt.close()