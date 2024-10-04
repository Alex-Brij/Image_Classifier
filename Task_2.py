# In this file please complete the following task:
#
# Task 2 [4] Basic evaluation
#
# Evaluate your classifiers. On your own, implement a method that will create a confusion matrix based on the provided
# classified data. Then implement methods that will output precision, recall, F-measure, and accuracy of your classifier
# based on your confusion matrix. Use macro-averaging approach and be mindful of edge cases. The template contains
# a range of functions you need to implement for this task.
#
# You can start working on this task immediately. Please consult at the very least Week 3 materials.
#
# You are expected to rely on solutions from Task_1_5 here! Do not reimplement kNN from scratch. You can ONLY rely
# on functions that were originally in the template in your final submission! Any functions you have created on your
# own and need here, must be defined here.

import Task_1_5 as Task_1_5
import Dummy
import numpy

# This function computes the confusion matrix based on the provided data.
#
# INPUT: classified_data   : a numpy arrays containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
# OUTPUT: confusion_matrix : the confusion matrix computed based on the classified_data.
#                            The order of elements MUST be the same as  in the classification scheme.
#                            The columns correspond to actual classes and rows to predicted classes.
#                            In other words, confusion_matrix[0] should be understood
#                            as the row of values predicted as Female, and [row[0] for row in confusion_matrix] as the
#                            column of values that were actually Female


def confusionMatrix(classified_data):
    # Intialise list of classes used to assing an entry in classified_data the correct indexes for row and column
    scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']

    # Intialise confusion matrix 2D array
    confusion_matrix = numpy.zeros((5, 5))

    # loops through rows in classifed data, gets the actaul and predicted classes for each entry
    for row in range(1, numpy.shape(classified_data)[0]):
        actual_class = classified_data[row][1]
        predicted_class = classified_data[row][2]

        # assings the correct row and column index to enter the corrected and predicted classes correctly into the confusion matrix
        actual_index = scheme.index(actual_class)
        predicted_index = scheme.index(predicted_class)

        # increments the confuion matrix at the right indexes to indicate a specfic combinaion of predicted and actaull classes
        confusion_matrix[predicted_index][actual_index] += 1

    return confusion_matrix


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
#
# INPUT: confusion_matrix : the confusion matrix computed based on the classified_data. The order of elements is
#                           the same as  in the classification scheme. The columns correspond to actual classes
#                           and rows to predicted classes.
# OUTPUT: a list of appropriate true positive, false positive or false
#         negative values per a given class, in the same order as in the classification scheme. For example, tps[1]
#         corresponds for TPs for Male class.


def computeTPs(confusion_matrix):
    # True positive is if predicted and actual are the same for each class.
    tps = []
    
    # loops through confusion matrix row by row and adds the true positive value (the number on the diagonal) to tps
    for row in range(numpy.shape(confusion_matrix)[0]):
        column = row
        tps.append(confusion_matrix[row][column])

    # print(f'true positive per class: {tps}')
    return tps


def computeFPs(confusion_matrix):
    # False positive if predicted to be a class but actually a different class 
    fps = []
    current_fps = 0

    # loops through confusion matrix row by row
    for row in range(numpy.shape(confusion_matrix)[0]):
        # for each row loops through collumn, if collumn index not the same same as row index (which would be a True Positive)
        # add the number of occurences of that FP to the variable for fps for the current class
        for column in range(numpy.shape(confusion_matrix)[1]):
            if row != column:
                current_fps += confusion_matrix[row][column]

        fps.append(current_fps)
        current_fps = 0

    # print(f'false positive per class: {fps}')
    return fps


def computeFNs(confusion_matrix):
    # False negative if actually in a class but predicted to be in a differnt class
    fns = []
    current_fns = 0

    # loops through confusion matrix column by column
    for column in range(numpy.shape(confusion_matrix)[1]):
        # for each column loops through row, if row index not the same same as column index (which would be a True Positive)
        # add the number of occurences of that FN to the variable for fns for the current class
        for row in range(numpy.shape(confusion_matrix)[0]):
            if column != row:
                current_fns += confusion_matrix[row][column]
                
        fns.append(current_fns)
        current_fns = 0

    # print(f'false negatives per class: {fns}')
    return fns


# These functions compute the evaluation measures based on the provided values. Not all measures use of all the values.
#
# INPUT: tps, fps, fns, data_size
#                       : the per-class true positives, false positive and negatives, and size of the classified data.
# OUTPUT: appropriate evaluation measures created using the macro-average approach.

def computeMacroPrecision(tps, fps, fns, data_size):
    precision = float(0)
    total_precision = 0

    # loops through number of true positves and false positives for each class and calculates the precison for each class
    for index in range(len(tps)):
        if (tps[index] + fps[index]) == 0:
            current_precision = 0
        else:
            current_precision = (tps[index] / (tps[index] + fps[index]))

        total_precision += current_precision

    # sums all the precisions and divides by the number of classes to find the macro_average precision
    precision = total_precision / len(tps)

    return precision


def computeMacroRecall(tps, fps, fns, data_size):
    recall = float(0)
    total_recall = 0

    # loops through number of true positves and false negatives for each class and calculates the recall for each class
    for index in range(len(tps)):
        if (tps[index] + fns[index]) == 0:
            current_recall = 0
        else:
            current_recall = (tps[index] / (tps[index] + fns[index]))

        total_recall += current_recall

    # sums all the recalls and divides by the number of classes to find the macro_average recall
    recall = total_recall / len(tps)

    return recall


def computeMacroFMeasure(tps, fps, fns, data_size):
    f_measure = float(0)
    total_f_measure = 0

    # loops through number of true positves and false postives and false negatives for each class to calcualte the f-measure for each class
    for index in range(len(tps)):
        if (((tps[index] + fps[index]) == 0) or ((tps[index] + fns[index]) == 0)):
            current_f_measure = 0
        else:
            current_precision = (tps[index] / (tps[index] + fps[index]))
            current_recall = (tps[index] / (tps[index] + fns[index]))
            if current_precision + current_recall == 0:
                current_f_measure = 0
            else:
                current_f_measure = (2 * current_precision * current_recall) / (current_precision + current_recall)


        total_f_measure += current_f_measure

    # sums all the f-measures and divides by the number of classes to find the macro_average f-measure
    f_measure = total_f_measure / len(tps)
    
    return f_measure


def computeAccuracy(tps, fps, fns, data_size):
    accuracy = float(0)
    # sum of true positves divided by the number of data entries
    accuracy = sum(tps) / data_size
    return accuracy


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.

# INPUT: classified_data   : a numpy array containing paths to images, actual classes and predicted classes.
#                            Please refer to Task 1 for precise format description.
#       confusion_func     : function to be invoked to compute the confusion matrix
#
# OUTPUT: computed measures

def evaluateKNN(classified_data, confusion_func=confusionMatrix):
    precision = float(-1)
    recall = float(-1)
    f_measure = float(-1)
    accuracy = float(-1)

    # calls the confusion_func to generate the confusion matrix from the csv data file of classifed data
    confusion_matrix = confusion_func(classified_data)

    # calculate the per class true positves, false postives and false negatives using the respective functions and the ocnfusion matrix
    tps = computeTPs(confusion_matrix)
    fps = computeFPs(confusion_matrix)
    fns = computeFNs(confusion_matrix)

    data_size = numpy.sum(confusion_matrix)

    # calculate the overeall prescion, recall, f-measure and accuracy using the respective functions and the macro-average method
    precision = computeMacroPrecision(tps, fps, fns, data_size)
    recall = computeMacroRecall(tps, fps, fns, data_size)
    f_measure = computeMacroFMeasure(tps, fps, fns, data_size)
    accuracy = computeAccuracy(tps, fps, fns, data_size)

    # once ready, we return the values
    return precision, recall, f_measure, accuracy


##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["classified_data"]}')
    classified_data = Task_1_5.readCSVFile(opts['classified_data'])
    print('Evaluating kNN')
    result = evaluateKNN(classified_data, eval(opts['cf']))
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
