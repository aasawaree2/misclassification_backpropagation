# misclassification_backpropagation
The criteria for classification algorithms are often stated in terms of maximizing overall classification accuracy (or minimizing MSE). It may instead be preferable to have constraints on misclassification rates for different classes. Suggested is a modification of the back-propagation algorithm for this task, evaluating performance on one of the two-class problems adapted from UCI datasets, examining accuracy as well as computational cost.

Existing backpropagation algorithm was modified to take into consideration the misclassification cost in the form of adapting new learning rate.

Train and Test without misclassification cost modification
TN, TP, FN, FP => 88 42 5 5
Total Misclassified data = 10
Accuracy = 92.86

Train and Test with misclassification cost modification
TN, TP, FN, FP => 88 45 2 5
Total Misclassified data = 7
Accuracy = 95.00
Total iterations - 126

References :-
1. https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
2. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.13.8285&rep=rep1&type=pdf
3. http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
4. https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
5. https://medium.com/themlblog/splitting-csv-into-train-and-test-data-1407a063dd74
6. https://ai.stackexchange.com/questions/4748/classifier-that-minimizes-false-positiveerror
7. https://www.verywellhealth.com/what-does-malignant-and-benign-mean-514240
