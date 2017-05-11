import numpy as np
import matplotlib.pyplot as plt



from decision_tree import DecisionTree
from random_forest import RandomForest




def accuracy_score(Y_true, Y_predict):

    s=0
    for i in range(len(Y_true)):
        if Y_true[i]==Y_predict[i]:
            s+=1
    return float(s)/len(Y_true)

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'data/SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    n, d = X.shape
    
    all_accuracies=[]
    all_accuracies_random_forest=[]

    for trial in range(1):

        idx = np.arange(n)
#        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]
        

        
        fold_shape=n//10

        k=0
        for j in range(10):
            print('epoch  '+str(j)+'  is running ....')
            Xtest=X[k:k+fold_shape]
            ytest=y[k:k+fold_shape]

            Xtrain=np.array(list(X[:k])+list(X[k+fold_shape:]))
            ytrain=np.array(list(y[:k])+list(y[k+fold_shape:]))
            
            k+=fold_shape
                        
            classifier = DecisionTree(100)

            classifier.fit(Xtrain, ytrain) 
                    
            y_pred = classifier.predict(Xtest)

            accuracy = accuracy_score(ytest, y_pred)
            all_accuracies.append(accuracy)
            
            #random_forest
            classifier_random_forest=RandomForest(10,100)
            
            classifier_random_forest.fit(Xtrain, ytrain)
            
            y_pred_random_forest=classifier_random_forest.predict(Xtest)[0]

            
            accuracy_random_forest=accuracy_score(ytest, y_pred_random_forest)
            all_accuracies_random_forest.append(accuracy_random_forest)
            

            
    all_accuracies=np.array(all_accuracies)


    # compute the training accuracy of the model
    meanDecisionTreeAccuracy = np.mean(all_accuracies)


    
    stddevDecisionTreeAccuracy = np.std(all_accuracies)
    meanLogisticRegressionAccuracy = 0
    stddevLogisticRegressionAccuracy = 0
    meanRandomForestAccuracy = np.mean(all_accuracies_random_forest)
    stddevRandomForestAccuracy = np.std(all_accuracies_random_forest)


    # make certain that the return value matches the API specification
    stats = np.zeros((3, 3))
    stats[0, 0] = meanDecisionTreeAccuracy
    stats[0, 1] = stddevDecisionTreeAccuracy
    stats[1, 0] = meanRandomForestAccuracy
    stats[1, 1] = stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats



if __name__ == "__main__":
    stats = evaluate_performance()
    print "Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")"
    print "Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")"
    print "Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")"

