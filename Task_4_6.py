# In this file please complete the following tasks:
#
# Task 4 [3] The curse of k
#
# Independent inquiry time! Picking the right number of neighbours k in the kNN approach is tricky.
# Find a way you could approach this more rigorously. In comments:
# •	state the name of the approach you could use,
# •	give a one-sentence explanation of the approach, and
# •	provide a reference to it (use Cardiff University Harvard style, DOI MUST BE PRESENT).
#
# The reference must be a handbook or peer-reviewed publication; a link to an online tutorial will not be accepted.
# Ensure that your resources are respectable and are not e.g., predatory journals.
#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.
# 
# 
# Correlation Matrix kNN 
# 
# This approach learns different k values for different test data points (as using the same k values led to sub optimal results)
# by using a reconstruction process between the training and testing data that uses a 1-norm reularizer (resulting in element-wise sparsity)
# to generate the different k values and a 2,1-norm regulaizer (to generate the row sparsity) to remove the impact of noisy data points.
# 
# SHICHAO ZHANG, XUELONG LI, MING ZONG, XIAOFENG ZHU, and DEBO CHENG. 2017. Learning k for kNN Classification. ACM Transactions on Intelligent Systems and Technology Volume 8 Issue 3 Article No.: 43pp 1–19. doi: 10.1145/2990508
# 
# 
#
#
# Task 6 [3] I can do better!
#
# Independent inquiry time! There are much better approaches out there for image classification.
# Your task is to find one, and using the comment section of your project, do the following:
# •	State the name of the approach
# •	Provide a permalink to a resource in the Cardiff University library that describes the approach
# •	Briefly explain how the approach you found is better than kNN in image classification (2-3 sentences is enough).
#  Focus on synthesis, not recall!
#
# You can start working on this task immediately. Please consult at the very least Week 2 materials.
# 
# 
# Convolutional Neural Network
# 
# https://librarysearch.cardiff.ac.uk/permalink/44WHELF_CAR/b7291a/cdi_springer_books_10_1007_978_981_99_7882_3
# 
# Convolutional Neural Networks are superior to KNN for image classfication for several reasons. Firstly in our case when finding distances between images with
# KNN we are treating the images as flat vectors of pixel values, without considering the spatial relationships between the pixels wheras CNN, using its 
# convolutional and pooling layers, gets the spatial structure of the image, which allows it to learn patterns and automaticaly extract features as well as understand
# the sturcture and context of the image. The initial layers detect the low level features like shapes/edges which are combined in later layers to allow more detailed  
# understanding of the image. If KNN was used with manual feature extraction this would be very time consuming whereas the automatic feature extraction of KNN would 
# be quicker and not rely on a person to do the work which is particulary useful as the amount of data being analysed increases.
# 
# 
