# In this file please complete the following task:
#
# Task 3 [6] Cross validation
#
# Evaluate your classifiers using the k-fold cross-validation technique covered in the lectures
# (use the training data only). Output their average precisions, recalls, F-measures and accuracies.
# You need to implement the validation yourself. Remember that folds need to be of roughly equal size.
# The template contains a range of functions you need to implement for this task.
#

# You are expected to rely on solutions from Task_1_5/Task_2 here! Do not reimplement kNN from scratch.
# You can ONLY rely on functions that were originally in the template in your final submission!
# Any functions you have created on your own in these files and need here, must be defined here.

import os
import Task_1_5 as Task_1_5
import Task_2 as Task_2
import Dummy
import numpy
from Task_1_5 import computeMeasure1,computeMeasure2,computeMeasure3,selfComputeMeasure1,selfComputeMeasure2


# This function takes the data for cross evaluation and returns training_data a list of lists s.t. the first element
# is the round number, second is the training data for that round, and third is the testing data for that round
#
# INPUT: training_data      : a numpy array was read from the training data csv (see parse_arguments function)
#        f                  : the number of folds to split the data into (which is also same as # of rounds)
# OUTPUT: folds             : a list of lists s.t. the first element is the round number, second is the numpy array
#                             representing the training data for that round, and third is the numpy array representing
#                             the testing data for that round
#                             You must make sure that the training and testing data are ready for use
#                             (e.g. contain the right headers already)

def splitDataForCrossValidation(training_data, f):
    # initilaises list of folds
    folds = []

    number_entries = (numpy.shape(training_data)[0] - 1)
    # works out base width of folds
    initial_fold_width = (number_entries // f)
    remainder = (number_entries % f)
    
    # creates list of fold widths
    splits = [initial_fold_width] * f

    # works out if base width of some folds will need to be increased by one to account for remainder when floor dividing
    # edits list of fold widths
    if remainder != 0:
        for i in range(remainder):
            splits[i] = splits[i] + 1

    start = 1
    round = 1

    # iterates through the folds adding the round, the subset of data for training and the subset of data for testing
    for width in splits:
        end = start + width

        # based on the of the current start and end indexes for the testing subset uses slices to get the headers 
        # as well as the testing subset, and also the training subset
        # adds the different parts of the partions together to given the arrays for testing and training
        new_testing_data = numpy.concatenate((training_data[:1], training_data[start:end]), axis=0)
        new_training_data = numpy.concatenate((training_data[:start], training_data[end:]), axis=0)

        folds.append([round, new_training_data, new_testing_data])

        start = end
        round += 1

    return folds


# In this function, please implement validation of the data that is produced by the cross evaluation function PRIOR to
# the addition of rows with the average measures.
#
# INPUT:  data              : the numpy array that was produced by the crossEvaluateKNN function BEFORE the
#                             addition of the rows with evaluation measures
#         f                 : number of folds to validate against
#
# OUTPUT: boolean value     : True if the data contains the header ["Path", "ActualClass", "PredictedClass","FoldNumber"]
#                             (there can be more column names, but at least these four at the start must be present)
#                             AND the values in the "Path" column (if there are any) are file paths
#                             AND the values in the "ActualClass" and "PredictedClass" columns
#                             (if there are any) are classes from the scheme
#                             AND the values in the "FoldNumber" column are integers in [0,f) range
#                             AND there are as many Path entries as ActualClass and PredictedClass and FoldNumber entries
#                             AND the number of entries per each integer in [0,f) range for FoldNumber are approximately
#                             the same (they can differ by at most 1)
#
#                             False otherwise

def validateDataFormat(data, f):

    classification_scheme = ['Female', 'Male', 'Primate', 'Rodent', 'Food']

    # counts how many validation checks are passed
    count = 0
    # First checks if correct collumn headers are present
    # then checks if pathsw are exist, classes are in the class scheme, numbers in fold number collumn are possible fold numbers and if
    # length of all data collumns are equivalent
    if ((data[0, 0] == 'Path') and (data[0, 1] == 'ActualClass') and (data[0, 2] == 'PredictedClass') and (data[0, 3] == 'FoldNumber')):
        if all(os.path.exists(path) for path in data[1:, 0]):
            count += 1
        if all(element in classification_scheme for element in data[1:, 1]):
            count += 1
        if all(element in classification_scheme for element in data[1:, 2]):
            count += 1
        if all(element in range(0, f + 1) for element in data[1:, 3]):
            count += 1
        if ((len(data[1:, 0]) == len(data[1:, 1])) and (len(data[1:, 1]) == len(data[1:, 2])) and (len(data[1:, 2]) == len(data[1:, 3]))):
            count += 1
        
        # finaly checks if for each fold number of data aentries are simmilar (within 1 of each other)
        check = True

        # creates list the size of the number of folds/rounds
        round_sizes = [0] * f # data[-1, 3]
        # counts number of data entries for each fold/round
        for entry in data[1:]:
            round_sizes[(entry[3] - 1)] += 1
        
        # checks that all amounts of data for each round differ by no more than 1 from the number of entries for the first round
        for index in range(len(round_sizes)):
            if (abs(round_sizes[0] - round_sizes[index]) <= 1):
                check == True
            else: 
                check == False
                break

        if check == True:
            count += 1

    # if all 6 validation checks are passed function returns True
    if count != 6:
        return False
    else:
        return True


# This function takes the classified data from each cross validation round and calculates the average precision, recall,
# accuracy and f-measure for them.
# Invoke either the Task 2 evaluation function or the dummy function here, do not code from scratch!
#
# INPUT: classified_data_list
#                           : a list of numpy arrays representing classified data computed for each cross validation round
#        evaluation_func    : the function to be invoked for the evaluation (by default, it is the one from
#                             Task_2, but you can use dummy)
# OUTPUT: avg_precision, avg_recall, avg_f_measure, avg_accuracy
#                           : average evaluation measures. You are expected to evaluate every classified data in the
#                             list and average out these values in the usual way.

def evaluateCrossValidation(classified_data_list, evaluation_func=Task_2.evaluateKNN):
    avg_precision = float(-1)
    avg_recall = float(-1)
    avg_f_measure = float(-1)
    avg_accuracy = float(-1)

    list_length = len(classified_data_list)
    total_precision = 0
    total_recall = 0
    total_f_measure = 0
    total_accuracy= 0

    # loops through the array lists in the list and adds the 4 measures for each array list to the variables storing the totals
    for array in classified_data_list:
        precision, recall, f_measure, accuracy = evaluation_func(array)
        total_precision += precision
        total_recall += recall
        total_f_measure += f_measure
        total_accuracy += accuracy

    # works out the average of each measure and returns
    avg_precision = total_precision / list_length
    avg_recall = total_recall / list_length
    avg_f_measure = total_f_measure / list_length
    avg_accuracy = total_accuracy / list_length


    return avg_precision, avg_recall, avg_f_measure, avg_accuracy


# In this task you are expected to perform cross-validation where f defines the number of folds to consider.
# "processed" holds the information from training data along with the following information: for each image,
# stated the id of the fold it landed in, and the predicted class it was assigned once it was chosen for testing data.
# After everything is done, we add the average measures at the end. The writing to csv is done in a different function.
# You are expected to invoke the Task 1 kNN classifier or the Dummy classifier here, do not implement these things
# from scratch!
#
# INPUT: training_data      : a numpy array that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours, to be passed to the kNN classifier
#        measure_func       : the function to be invoked to calculate similarity/distance
#        similarity_flag    : a boolean value stating that the measure above used to produce the values is a distance
#                             (False) or a similarity (True)
#        knn_func           : the function to be invoked for the classification (by default, it is the one from
#                             Task_1_5, but you can use dummy)
#        split_func         : the function used to split data for cross validation (by default, it is the one above)
#        f                  : number of folds to use in cross validation
# OUTPUT: processed       : a list of lists which expands the training_data with columns stating the fold number to
#                             which a given image was assigned and the predicted class for that image; and with rows
#                             that contain the average evaluation measures (see the h and v variables)
#                             IF validation of the processed variable fails (prior to addition of evaluation measures),
#                             return only the header!
# Again, please remember to have a look at the Dummy file!
def crossEvaluateKNN(training_data, k, measure_func, similarity_flag, f, knn_func=Task_1_5.kNN,
                     split_func=splitDataForCrossValidation):
    

    # This adds the header
    processed = numpy.array([['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']])
    avg_precision = -1.0
    avg_recall = -1.0
    avg_fMeasure = -1.0
    avg_accuracy = -1.0

    # gives the training data array, splits it into multiple rounds of knn image classification and returns list of lists 
    # each list containg a round of knn with arrays for the training data and testing data for that round 
    folds = split_func(training_data, f)

    # list of classifed data arrays- image, actual class, predicted class, fold number
    classified_list = []
    # list of classified data arrays- image, actual class, predicted class
    # without round number- used for evaluating class measures
    classified_list_no_round = []

    for fold in folds:
        # for each fold/round calls the knn function from task 1 to complete the image clasification
        # folds[1] is training data array, folds[2] is testing data array (data to classify)
        # returns classified_data array- image, actaual class, predicted class
        classified_data = knn_func(fold[1], k, measure_func, similarity_flag, fold[2])
        # without adding round information adds to list of lists
        classified_list_no_round.append(classified_data)

        # adds new collumn containing fold numbers to classified data
        new_column = numpy.empty(classified_data.shape[0] - 1, dtype=object)
        new_column[0:] = fold[0]
        # stacks new column of fold numbers side by side with classified data
        classified_data_with_round = numpy.hstack((classified_data[1:], new_column.reshape(-1, 1)))
        
        # with round information adds classified data to list of lists
        classified_list.append(classified_data_with_round)


    # list of arrays that contain classified data including round number are all concatenated into one big array
    # then added to the processed array containing the headers
    data = numpy.concatenate(classified_list, axis=0)
    processed = numpy.concatenate((processed, data), axis=0)
  

    if validateDataFormat(processed, f):

        # evaluates measures for data
        avg_precision, avg_recall, avg_fMeasure, avg_accuracy = evaluateCrossValidation(classified_list_no_round)

        # # The measures are now added to the end. You should invoke validation BEFORE this step.
        h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
        v = [avg_precision, avg_recall, avg_fMeasure, avg_accuracy]

        processed = numpy.append(processed, [h], axis=0)
        processed = numpy.append(processed, [v], axis=0)

        # print(processed)
        return processed

    else:
        return numpy.array([['Path', 'ActualClass', 'PredictedClass', 'FoldNumber']])




##########################################################################################
# You should not need to modify things below this line - it's mostly reading and writing #
# Be aware that error handling below is...limited.                                       #
##########################################################################################


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parseArguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.readCSVFile(opts['training_data'])
    print('Evaluating kNN')
    result = crossEvaluateKNN(training_data, opts['k'], eval(opts['measure']), opts['simflag'], opts['f'],
                              eval(opts['al']), eval(opts['sf']))
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.writeCSVFile(out, result)


if __name__ == '__main__':
    main()
