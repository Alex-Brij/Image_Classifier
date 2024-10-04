import argparse
import csv
import distutils.util
import distutils
import math
import os
import Dummy
import numpy

from PIL import Image
from scipy.spatial import distance 
import os

# Task 1 [10] My first not-so-pretty image classifier
#
# By using the kNN approach and three distance or similarity measures, build image classifiers.
#•	You must implement the kNN approach yourself
#•	You must invoke the distance or similarity measures from libraries (it is fine to invoke different measures from
#   one library). Non-trivial adjustments to a library-invoked measure do not meet the requirements!
#•	Histogram-based measures are not allowed
#•	Jaccard distances/similarities are not allowed
#•	You can use between 0 and 3 distance measures and between 0 and 3 similarity measures
#   (there is no requirement that at least one of each kind should be present)
#
# The classifier is expected to use only one measure at a time and take information as to which one to invoke at a given
# time as input. The template contains a range of functions you must implement and use appropriately for this task.
#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.

# Task 5 [4] Similarities
#
# Independent inquiry time! In Task 1, you were instructed to use libraries for image similarity measures.
# Pick two of the three measures you have used and implement them yourself.
# You are allowed to use libraries to e.g., calculate the root, power, average or standard deviation of some set
# (but, for example, numpy.linalg.norm is not permitted).
# The template contains a range of functions you need to implement for this task.
#
# Disclaimer: if you decide to implement MSE, do not implement RMSE (and vice versa)
#
# You can start working on this task immediately. Please consult at the very least Week 1 materials.


# Please replace with your student id, including the "c" at the beginning!!!
student_id = 'c21044707'

# This is the classification scheme you should use for kNN
classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']


# In this function, please implement validation of the data that is supplied to or produced by the kNN classifier.
#
# INPUT:  data              : numpy array that was read from the training data or data to classify csv
#                             (see parse_arguments function) or produced by the kNN function
#         predicted         : a boolean value stating whether the "PredictedClass" column should be present
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass"] if predicted variable
#                             is False and ["Path", "ActualClass", "PredictedClass"] if it is True
#                             (there can be more column names, but at least these three at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" column (if there are any) are classes from scheme
#                             AND (if predicted is True) the values in the "PredictedClass" column (if there are any)
#                             are classes from scheme
#                             AND there are as many Path entries as ActualClass (and PredictedClass, if predicted
#                             is True) entries
#
#                             False otherwise

def validateDataFormat(data, predicted):
    # If there is no Actual Class collumn, validates the data, checks collumn nanmes, if path is a path in the system, 
    # if the class is from the classification scheme and if the number of path entries matches number of class entries
    if predicted == False:
        count = 0

        if (data[0, 0] == 'Path') and (data[0, 1] == 'ActualClass'):
            if all([os.path.exists(path) for path in data[1:, 0]]):
                count += 1
            if all(element in classification_scheme for element in data[1:, 1]):
                count += 1
            if len(data[1:, 0]) == len(data[1:, 1]):
                count += 1
        
        if count != 3:
            return False
        else:
            return True
        
    # Does the same checks but also checks for the predicted class collumn
    if predicted == True:
        count = 0
        if ((data[0, 0] == 'Path') and (data[0, 1] == 'ActualClass') and (data[0, 2] == 'PredictedClass')):
            if all(os.path.exists(path) for path in data[1:, 0]):
                count += 1
            if all(element in classification_scheme for element in data[1:, 1]):
                count += 1
            if all(element in classification_scheme for element in data[1:, 2]):
                count += 1
            if ((len(data[1:, 0]) == len(data[1:, 1])) and (len(data[1:, 1]) == len(data[1:, 2]))):
                count += 1

        if count != 4:
            return False
        else:
            return True


    formatCorrect = False

    return formatCorrect


# This function does reading and resizing of an image located in a give path on your drive.
# DO NOT REMOVE ANY COLOURS. DO NOT MODIFY PATHS. DO NOT FLATTEN IMAGES.
#
# INPUT:  imagePath         : path to image. DO NOT MODIFY - take from the file as-is. Things like appending "..\"
#                             to the file path within the code are not permitted.
#         width, height     : width and height dimensions to which you are asked to resize your image
#
# OUTPUT: image             : numpy array representing the read and resized image in RGB format
#                             (empty if the image is not found at a given path).
#                             Removing colour channels (e.g. transforming array to grayscale) or flattening the image
#                             ARE NOT PERMITTED.
#

def readAndResize(image_path, width=60, height=30):
    # loads image from path and resizes to width and height arguments
    loaded_image = Image.open(image_path)
    resized_iamge = loaded_image.resize((width, height))

    # converts image to numpy array
    image = numpy.asarray(resized_iamge)
    return image


# These functions compute the distance or similarity value between two images according to a particular
# similarity or distance measure. Return nan if images are empty. These three measures must be
# computed by libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two numpy arrays representing images in RGB formats. Do NOT presume a particular height
#                             or width! If you need images flattened or in grayscale or in any other format, then these
#                             manipulations will need to take place WITHIN the computeMeasure functions.
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def computeMeasure1(image1, image2):
    # Euclidean Distance

    # flattens images to 1D arrays
    flat_image_1 = image1.ravel()
    flat_image_2 = image2.ravel()

    # if either image array is empy returns nan
    if (((numpy.shape(image1)[0]) == 0) or ((numpy.shape(image2)[0]) == 0)):
        value = float('nan')
    # else calculates euclidean distance between the two images using scipy function
    else:
        value = distance.euclidean(flat_image_1, flat_image_2)

    # print(f'value of distance from auto euclidean: {value}')
    return value


def computeMeasure2(image1, image2):
    # Manhattan Distance

    # flattens images to 1D arrays
    flat_image_1 = image1.ravel()
    flat_image_2 = image2.ravel()

    # if either image array is empy returns nan
    if (((numpy.shape(image1)[0]) == 0) or ((numpy.shape(image2)[0]) == 0)):
        value = float('nan')
    # else calculates manhattan distance between the two images using scipy function
    else:
        value = distance.cityblock(flat_image_1, flat_image_2)
   
    # print(f'value of distance from auto manhattan {value}')
    return value


def computeMeasure3(image1, image2):
    # Cosine Distance

    # flattens images to 1D arrays
    flat_image_1 = image1.ravel()
    flat_image_2 = image2.ravel()

    # if either image array is empy returns nan
    if (((numpy.shape(image1)[0]) == 0) or ((numpy.shape(image2)[0]) == 0)):
        value = float('nan')
    # else calculates cosine distance between the two images using scipy function
    else:
        value = distance.cosine(flat_image_1, flat_image_2)

    # print(f'value of distance from auto cosine: {value}')
    return value


# These functions compute the distance or similarity value between two images according to a particular similarity or
# distance measure. Return nan if images are empty. As name suggests, selfComputeMeasure 1 has to be your own
# implementation of the measure you have used in computeMeasure1 (same for 2). These two measures cannot be computed by
# libraries according to portfolio requirements.
#
# INPUT:  image1, image2    : two numpy arrays representing images in RGB formats. Do NOT presume a particular height
# #                           or width! If you need images flattened or in grayscale or in any other format, then these
# #                           manipulations will need to take place WITHIN the computeMeasure functions.
#
# OUTPUT: value             : the distance or similarity value between image1 and image2 according to a chosen approach.
#                             Defaults to nan if images are empty.
#

def selfComputeMeasure1(image1, image2):
    # Euclidean Distance

    # compute euclidean distance using pixel-wise calculations for each channel (RGB)

    # if either image array is empy returns nan
    if (((numpy.shape(image1)[0]) == 0) or ((numpy.shape(image2)[0]) == 0)):
        value = float('nan')

    else:
        total_distance_squared = 0
        # loops through each layer (RGB) of the image
        for layer in range(3):
            # sets current_layer 1 & 2 to the corresponding 2D array for the current RGB layer
            current_layer_1 = image1[:, :, layer]
            current_layer_2 = image2[:, :, layer]

            # iterates through every pixel for the current layer from image 1
            for i in range(numpy.shape(current_layer_1)[0]):
                for j in range(numpy.shape(current_layer_1)[1]):
                    # calculaes the difference squared between the pixel from image 1 and the coresponding pixel from image 2 and adds to total
                    distance_squared = ((current_layer_1[i, j] - current_layer_2[i, j]) ** 2)
                    total_distance_squared += distance_squared

        # Finds square root of total squared distance between both images
        value = math.sqrt(total_distance_squared)

    # print(f'value of distance from custom euclidean{value}')
    return value


def selfComputeMeasure2(image1, image2):
    # Manhattan Distance

    # compute manhattan distance using pixel-wise calculations for each channel (RGB)

    # if either image array is empy returns nan
    if (((numpy.shape(image1)[0]) == 0) or ((numpy.shape(image2)[0]) == 0)):
        value = float('nan')

    else:
        total_distance = 0
        # loops through each layer (RGB) of the image
        for layer in range(3):
            # sets current_layer 1 & 2 to the corresponding 2D array for the current RGB layer
            current_layer_1 = image1[:, :, layer]
            current_layer_2 = image2[:, :, layer]

            # iterates through every pixel for the current layer from image 1
            for i in range(numpy.shape(current_layer_1)[0]):
                for j in range(numpy.shape(current_layer_1)[1]):
                    # calculaes the absoloute distance between the pixel from image 1 and the coresponding pixel from image 2 and adds to total
                    distance = abs((current_layer_1[i, j] - current_layer_2[i, j]))
                    total_distance += distance

        value = total_distance

    # print(f'value of distance from custom manhatan {value}')
    return value


# This function is supposed to return a dictionary of classes and their occurrences as taken from k nearest neighbours.
#
# INPUT:  measure_classes   : a list of lists that contain two elements each - a distance/similarity value
#                             and class from scheme
#         k                 : the value of k neighbours
#         similarity_flag   : a boolean value stating that the measure used to produce the values above is a distance
#                             (False) or a similarity (True)
# OUTPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
def getClassesOfKNearestNeighbours(measures_classes, k, similarity_flag):
    # Intiliases dictionary to store frequency with which each class appears in k nearest neighbours
    nearest_neighbours_classes = {'Female':0, 'Male':0, 'Primate':0, 'Rodent':0, 'Food':0}

    # print(measures_classes)

    # Sorts list of lists based on distance/similarity in accending/decending order 
    if similarity_flag == False:
        accending_list = sorted(measures_classes, key=lambda x: x[0])
    elif similarity_flag == True:
        accending_list = sorted(measures_classes, key=lambda x: x[0], reverse=True)

    # print(accending_list)
    # loops through list of lists and increments classes stored in dictionary every time the class appears in the k nearest neighbours
    for sub_list in accending_list[0:k]:
        nearest_neighbours_classes[sub_list[1]] += 1
        
    # print(nearest_neighbours_classes)
    return nearest_neighbours_classes


# Given a dictionary of classes and their occurrences, returns the most common class. In case there are multiple
# candidates, it follows the order of classes in the scheme. The function returns empty string if the input dictionary
# is empty, does not contain any classes from the scheme, or if all classes in the scheme have occurrence of 0.
#
# INPUT: nearest_neighbours_classes
#                           : a dictionary that, for each class in the scheme, states how often this class
#                             was in the k nearest neighbours
#
# OUTPUT: winner            : the most common class from the classification scheme. In case there are
#                             multiple candidates, it follows the order of classes in the scheme. Returns empty string
#                             if the input dictionary is empty, does not contain any classes from the scheme,
#                             or if all classes in the scheme have occurrence of 0
#

def getMostCommonClass(nearest_neighbours_classes):
    # if dict empty or all keys = 0 return empy string  
    if (bool(nearest_neighbours_classes) == False) and (all(value == 0 for value in nearest_neighbours_classes.values())):
        winner = ''
    # else orders the dictionary, first by which class appears the most and secondly (if two classes draw) by the order of classes in the classification scheme
    # then picks the first class (the one that appears the most in the k nearest neighbours and hence the class assigned to the image)
    else:
        sorted_classes = sorted(nearest_neighbours_classes.items(), key=lambda x: (-x[1], list(nearest_neighbours_classes.keys()).index(x[0])))
        winner = ((sorted_classes[0])[0])


    return winner


# In this function I expect you to implement the kNN classifier. You are free to define any number of helper functions
# you need for this! You need to use all of the other functions in the part of the template above.
#
# INPUT:  training_data       : a numpy array that was read from the training data csv
#         k                   : the value of k neighbours
#         measure_func        : the function to be invoked to calculate similarity/distance (any of the above)
#         similarity_flag     : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#         data_to_classify    : a numpy array  that was read from the data to classify csv;
#                             this data is NOT be used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
#     most_common_class_func  : the function to be invoked to find the most common class among the neighbours
#                             (by default, it is the one from above)
# get_neighbour_classes_func  : the function to be invoked to find the classes of nearest neighbours
#                             (by default, it is the one from above)
#         read_func           : the function to be invoked to find to read and resize images
#                             (by default, it is the one from above)
#  OUTPUT: classified_data    : a numpy array which expands the data_to_classify with the results on how your
#                             classifier has classified a given image.
#                             IF the training_data or data_to_classify is empty OR
#                             training_data, data_to_classify, or produced classified_data fail validation,
#                             the returned array contains ONLY the header row


def kNN(training_data, k, measure_func, similarity_flag, data_to_classify,
        most_common_class_func=getMostCommonClass, get_neighbour_classes_func=getClassesOfKNearestNeighbours,
        read_func=readAndResize):
    
    # Initilaises the numpy array with the header list
    classified_data = numpy.array([['Path', 'ActualClass', 'PredictedClass']])
    
    # validates both the training data and testing data to make sure its in the right format 
    if (validateDataFormat(training_data, False) and validateDataFormat(data_to_classify, False)):
        # loops through all pieces of testing data
        for i in range(1, (numpy.shape(data_to_classify)[0])):
            # for each image in testing data initilises a measure_classes list 
            # reads the image in using the read and resize function and stores it as image_testing
            # stores that images class in image_class_testing
            measure_classes_list = []
            image_path_testing = data_to_classify[i, 0]
            image_testing = read_func(image_path_testing)
            image_class_testing = data_to_classify[i, 1]
            # print(f'TESTING photo: {i}')

            # for each testing image loops thorugh every training image
            for j in range(1, (numpy.shape(training_data)[0])):
                # for each training image reads the image in using the read and resize function and stores it as image_trainingt
                # stores that images class in image_class_testing
                image_path_training = training_data[j, 0]
                image_training = read_func(image_path_training)
                image_class_training = training_data[j, 1]
                # print(f'training photo: {j}')
                
                # uses the specified measure function to compare the distance/similarity between the testing and training image
                # and adds the calculated distance and the training images' class to the measure_classes list
                distance = measure_func(image_testing, image_training)
                measure_classes_list.append([distance, image_class_training])
            
            # with list of all distances and coresponing classes between the testing image and all the training images,
            # finds k nearest neighbours of testing image using get neighbour classes function
            nearest_neighbours = get_neighbour_classes_func(measure_classes_list, k, similarity_flag)
           
            # finds the wining preidcted class using most common classes funciton and appends it alongisde 
            # the actual image class and the image path to the classified data array 
            winning_class = most_common_class_func(nearest_neighbours)
            next_row = [image_path_testing, image_class_testing, winning_class]
            classified_data = numpy.vstack([classified_data, next_row]) 

            # checks if classified data matches valid format before returning it
            if not validateDataFormat(classified_data, True):
                classified_data = numpy.array([['Path', 'ActualClass', 'PredictedClass']])


    return classified_data


##########################################################################################
# Do not modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.

def main():
    opts = parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = readCSVFile(opts['training_data'])
    data_to_classify = readCSVFile(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    print(opts['simflag'])
    result = kNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], data_to_classify,
                 eval(opts['mcc']), eval(opts['gnc']), eval(opts['rrf']))
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        writeCSVFile(out, result)


# Straightforward function to read the data contained in the file "filename"
def readCSVFile(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return numpy.array(lines)


# Straightforward function to write the data contained in "lines" to a file "filename"
def writeCSVFile(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#                         (needed in Tasks 1, 2, 3 and 5)
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -measure        : function to compute a given similarity/distance measure
#       -simflag        : flag telling us whether the above measure is a distance (False) or similarity (True)
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#                         (needed in Tasks 1, 3 and 5)
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#                         (needed in Tasks 1, 2, 3 and 5)
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#                         (needed in Tasks 1, 2, 3 and 5)
#       mcc, gnc, rrf, vf,cf,sf,al
#                       : staff variables, do not use
#
def parseArguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-m', '--measure')
    parser.add_argument('-s', '--simflag', type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('-train', type=str)
    parser.add_argument('-test', type=str)
    parser.add_argument('-classified', type=str)
    parser.add_argument('-mcc', default="getMostCommonClass")
    parser.add_argument('-gnc', default="getClassesOfKNearestNeighbours")
    parser.add_argument('-rrf', default="readAndResize")
    parser.add_argument('-cf', default="confusionMatrix")
    parser.add_argument('-sf', default="splitDataForCrossValidation")
    parser.add_argument('-al', default="Task_1_5.kNN")
    params = parser.parse_args()

    opt = {'k': params.k,
           'f': params.f,
           'measure': params.measure,
           'simflag': params.simflag,
           'training_data': params.train,
           'data_to_classify': params.test,
           'classified_data': params.classified,
           'mode': params.unseen,
           'mcc': params.mcc,
           'gnc': params.gnc,
           'rrf': params.rrf,
           'cf': params.cf,
           'sf': params.sf,
           'al': params.al
           }
    return opt


if __name__ == '__main__':
    main()
