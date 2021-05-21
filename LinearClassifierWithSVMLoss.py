import numpy as np
import csv
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def predict(xsample, W):

    s = []
    # TODO - Application 3 - Step 2 - compute the vector with scores (s) as the product between W and xsample
    s = W.dot(xsample)

    return s
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 3 - Step 3 - The function that compute the loss for a data point
def computeLossForASample(s, labelForSample, delta):

    loss_i = 0
    syi = s[labelForSample]  # the score for the correct class corresonding to the current input sample based on the label yi

    # TODO - Application 3 - Step 3 - compute the loss_i
    for idx, sj in enumerate(s):
        if idx != labelForSample:
            loss_i = loss_i + max(0,sj-syi+delta)

    return loss_i
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
# TODO - Application 3 - Step 4 - The function that compute the gradient loss for a data point
def computeLossGradientForASample(W, s, currentDataPoint, labelForSample, delta):

    dW_i = np.zeros(W.shape)  # initialize the matrix of gradients with zero
    syi = s[labelForSample]   # establish the score obtained for the true class

    for j, sj in enumerate(s):
        dist = sj - syi + delta

        if j == labelForSample:
            continue

        if dist > 0:
            dW_i[j] = currentDataPoint
            dW_i[labelForSample] = dW_i[labelForSample] - currentDataPoint

    return dW_i
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():

    # Input points in the 4 dimensional space
    x_train = np.array([[1, 5, 1, 4],
                        [2, 4, 0, 3],
                        [2, 1, 3, 3],
                        [2, 0, 4, 2],
                        [5, 1, 0, 2],
                        [4, 2, 1, 1]])

    # Labels associated with the input points
    y_train = [0, 0, 1, 1, 2, 2]

    # Input points for prediction
    x_test = np.array([[1, 5, 2, 4],
                       [2, 1, 2, 3],
                       [4, 1, 0, 1]])

    # Labels associated with the testing points
    y_test = [0, 1, 2]

    # The matrix of wights
    W = np.array([[-1, 2,  1, 3],
                  [ 2, 0, -1, 4],
                  [ 1, 3,  2, 1]])


    delta = 1               # margin
    step_size = 0.01        # weights adjustment ratio


    loss_L = 0
    dW = np.zeros(W.shape)
    prev_loss = 100

    # TODO - Application 3 - Step 2 - For each input data...
    for idx, xsample in enumerate(x_train):

        # TODO - Application 3 - Step 2 - ...compute the scores s for all classes (call the method predict)
        s = predict(xsample,W)



        # TODO - Application 3 - Step 3 - Call the function (computeLossForASample) that
        #  compute the loss for a data point (loss_i)
        loss_i = computeLossForASample(s, y_train[idx], delta)



        # Print the scores - Uncomment this
        print("Scores for sample {} with label {} is: {} and loss is {}".format(idx, y_train[idx], s, loss_i))



        # TODO - Application 3 - Step 4 - Call the function (computeLossGradientForASample) that
        #  compute the gradient loss for a data point (dW_i)
        dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)



        # TODO - Application 3 - Step 5 - Compute the global loss for all the samples (loss_L)
        loss_L += loss_i



        # TODO - Application 3 - Step 6 - Compute the global gradient loss matrix (dW)
        dW += dW_i


    # TODO - Application 3 - Step 7 - Compute the global normalized loss
    loss_L = loss_L/len(y_train)
    print("The global normalized loss is:",'%.3f'%loss_L)


    # TODO - Application 3 - Step 8 - Compute the global normalized gradient loss matrix
    dW = dW/len(y_train)


    # TODO - Application 3 - Step 9 - Adjust the weights matrix
    W = W - step_size * dW

    # TODO - Exercise 6
    # i=1
    # while loss_L > 0.001:
    #     loss_L = 0
    #     for idx, xsample in enumerate(x_train):
    #         s = predict(xsample, W)
    #         loss_i = computeLossForASample(s, y_train[idx], delta)
    #         dW_i = computeLossGradientForASample(W, s, x_train[idx], y_train[idx], delta)
    #         dW += dW_i
    #         dW = dW / len(y_train)
    #         loss_L += loss_i
    #         loss_L = loss_L / len(y_train)
    #         i += 1
    #         print("Weights",W)
    #         W = W - step_size * dW
    #
    # print("The global normalized loss is:",'%.4f'%loss_L)
    # print("Necessary steps:",i)


    # TODO - Exercise 7 - After solving exercise 6, predict the labels for the points existent in x_test variable
    #  and compare them with the ground truth labels. What is the system accuracy?
    W =np.array([[-0.84998457,  2.55000309,  0.77402778,  3.26202315 ],
                 [ 0.99364818, -0.35967592,  0.15197222,  3.9706389, ],
                 [ 1.85633638,  2.80967283,  1.074,       0.76733795]])
    correctPredicted = 0
    predicts = 0
    for idx, xsample in enumerate(x_test):
        s = predict(xsample,W)
        indice = np.argmax(s)
        predicts += 1
        if indice == y_test[idx]:
            correctPredicted += 1

    accuracy = (correctPredicted/predicts)*100
    print("Accuracy for test = {}%".format(accuracy))

    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
