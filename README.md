# Handwritten_Digit
This project takes hand written digits into account and based upon training,it recognizes Test digits. 
To recognize handwritten digits in python we used machine learning library called scikit learn and applied knn algorithm. KNN (k nearest number) algo calculates the distance of a new data point to all other training data points.  It then selects the K-nearest data points, where K can be any integer  . Finally it assigns the data point to the class to which the majority of the K data points belong. This algo is used to test digits after training. To train the program we used Mnist data set and divided it into two parts one for training and other for testing  . After that we divided training set digits into 28*28 pixel and based upon the intensity of darkness of each box we assigned it a number 0 to 255 where 0 is completely blank and 255 as completely dark. Then by comparing that value to the label ,we trained the program and then tested it with help of knn algo.  
