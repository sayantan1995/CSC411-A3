import pickle
import numpy as np
import numpy.random as rnd 
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import sklearn.utils as utils
import bonnerlib2


# Q1 (a)

Xtrain, ttrain = datasets.make_moons(n_samples = 200, noise = 0.2)
Xtest, ttest = datasets.make_moons(n_samples = 10000, noise = 0.2)
colors = np.array(['r', 'b'])
plt.figure()
plt.suptitle('Figure 1, Question 1 (a): Moons training data')
plt.scatter(Xtrain[:,0], Xtrain[:,1], color = colors[ttrain], s = 5)
plt.figure()
plt.suptitle('Figure 2, Question 1(a): Moons test data')
plt.scatter(Xtest[:,0], Xtest[:,1], color = colors[ttest], s = 2)


# Q1 (b)

def fitMoons():
    plt.figure()
    plt.suptitle('Figure 3, Question 1(b): Contour plots of various training sessions')
    errMin = np.inf

    for i in range(9):
        clf = MLPClassifier(hidden_layer_sizes = [3], 
                            max_iter = 10000,
                            activation = 'tanh',
                            solver = 'sgd',
                            tol = 10.0**(-20),
                            learning_rate_init = 0.01)

        clf.fit(Xtrain, ttrain)
        err = 1 - clf.score(Xtest, ttest)
        print('neural net {}: test error = {}'.format(i + 1, err))
        plt.subplot(3, 3, i + 1)
        ax = plt.gca()
        plt.axis('off')
        plt.scatter(Xtrain[:,0], Xtrain[:,1], color = colors[ttrain], s = 2)
        bonnerlib2.dfContour(clf, ax)

        if err < errMin:
            errMin = err
            clfBest = clf


    plt.figure()
    plt.suptitle('Figure 4, Question 1(b): contour plot for best training session')
    ax = plt.gca()
    plt.scatter(Xtrain[:,0], Xtrain[:,1], color = colors[ttrain], s = 5)
    bonnerlib2.dfContour(clfBest, ax)
    print('\nMinimum test error = {}'.format(errMin))

print('\n')
print('Question 1(a).')
print('---------')
fitMoons()



# Q3 (a) 

def displaySample(N, data):
    M = int(np.ceil(np.sqrt(N)))
    m = int(np.sqrt(np.size(data[0])))
    sample = utils.resample(data, n_samples = N, replace= False)

    for i in range(0,N):
        x = sample[i]
        y = np.reshape(x, (m, m))
        plt.subplot(M, M, i + 1)
        plt.axis('off')
        plt.imshow(y, cmap = 'Greys', interpolation = 'nearest')


def flatten(data):
    K = len(data)
    X = np.vstack(data)
    N = np.shape(X)[0]
    t = np.zeros(N, dtype = 'int')
    m1 = 0
    m2 = 0

    for i in range(0, K):
        m = np.shape(data[i])[0]
        m2 += m
        t[m1:m2] = i
        m1 += m

    return X, t

with open('mnist.pickle','rb') as f:
    data = pickle.load(f)
  
Xtrain = data['training']
Xtest = data['testing']
Xtrain, Ytrain = flatten(Xtrain)
Xtest, Ytest = flatten(Xtest)
Xtrain, Ytrain = utils.shuffle(Xtrain, Ytrain)


# Q3 (b):

scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# Q3 (c) 

plt.figure()
plt.suptitle('Question 3(c): some normalized MNIST digits')
displaySample(16, Xtest)


# Q3 (d) + (e)

clf = MLPClassifier(solver = 'sgd',
                    alpha = 0.2,
                    tol = 0.0,
                    activation = 'tanh',
                    batch_size = 200,
                    learning_rate_init = 0.1,
                    max_iter = 5,
                    momentum = 0.5,
                    hidden_layer_sizes = [100],
                    warm_start = True)

print('\n')
print('Question 3 (e).')
print('---------')

for n in range(50):
    clf.fit(Xtrain, Ytrain)
    errTrain = 1 - clf.score(Xtrain, Ytrain)
    errTest = 1 - clf.score(Xtest, Ytest)
    print('Iteration {}: Training error = {:.2%}, Test error = {:.2%}'.format(n+1, errTrain, errTest))


# Q3 (f) 

clfBatch = MLPClassifier(solver = 'sgd',
                         alpha = 0.2,
                         tol = 0.0,
                         activation = 'tanh',
                         batch_size = 60000,
                         learning_rate_init = 0.1,
                         max_iter = 5,
                         momentum = 0.5,
                         hidden_layer_sizes = [100],
                         warm_start = True)

errTrainList = []
errTestList = []

for n in range(1000):
    clfBatch.fit(Xtrain, Ytrain)
    errTrain = 1 - clfBatch.score(Xtrain, Ytrain)
    errTest = 1 - clfBatch.score(Xtest, Ytest)
    print ('Iteration {}: Training error = {:.2%}, Test error = {:.2%}'.format(n+1, errTrain, errTest))
    errTrainList.append(errTrain)
    errTestList.append(errTest)

plt.figure()
plt.plot(errTrainList)
plt.plot(errTestList)
plt.suptitle('Figure 5, Question 3: training and test error in batch mode')
plt.xlabel('training iterations')
plt.ylabel('error')

plt.figure()
plt.plot(errTestList[-500:])
plt.suptitle('Figure 6, Question 3: test error during last 500 iterations of batch training')
plt.xlabel('training iterations')
plt.ylabel('error')


# Q4 (a) 

def predict(X, W1, W2, b1, b2):
    b1 = np.reshape(b1, [1,-1])
    H1 = np.tanh(np.matmul(X,W1) + b1)
    b2 = np.reshape(b2, [1, -1])
    Z2 = np.matmul(H1, W2) + b2
    expZ2 = np.exp(Z2)
    U = np.sum(expZ2, axis = 1)
    U = np.reshape(U, [-1, 1])
    Y = expZ2/U
    return H1, Y

print('\n')
print('Question 4(b).')
print('---------')
W = clf.coefs_
b = clf.intercepts_
H1, Y1 = predict(Xtest, W[0], W[1], b[0], b[1])
Y2 = clf.predict_proba(Xtest)
diff = np.sum((Y1 - Y2)**2)
print('total squared error = {}'.format(diff))


# Q4 (c)

def gradient(H, Y, T):
    N,_ = np.shape(H)
    Err = (Y - T)/N
    DW = np.matmul(np.transpose(H), Err)
    Db = np.sum(Err, axis = 0)
    return DW, Db


# Q4 (d)

def loss(T, Y):
    N, _ = np.shape(Y)
    return -np.sum(T * np.log2(Y))/N


def accuracy(T, Y):
    return np.mean(np.argmax(T, axis = 1) == np.argmax(Y, axis = 1))

Ntrain, M0 = np.shape(Xtrain)
Ntest, M0 = np.shape(Xtest)
M1 = 100
M2 = 10

Ttrain = np.zeros([Ntrain, M2])
Ttest = np.zeros([Ntest, M2])
Ttrain[range(Ntrain), Ytrain] = 1
Ttest[range(Ntest), Ytest] = 1 


def bgd(W1, b1, lrate, sigma, K):
    _, M1 = np.shape(W1)
    _, M2 = np.shape(Ttrain)
    W2 = sigma * rnd.randn(M1, M2)
    b2 = np.zeros([M2])

    lossList = []
    errTrainList = []
    errTestList = []

    for k in range(K):
        H, Y = predict(Xtrain, W1, W2, b1, b2)
        DW2, Db2 = gradient(H, Y, Ttrain)
        W2 = W2 - lrate * DW2
        b2 = b2 - lrate * Db2

        if np.mod(k, 5) == 0:
            lossTrain = loss(Ttrain, Y)
            errTrain = 1 - accuracy(Ttrain, Y)
            H, Y = predict(Xtest, W1, W2, b1, b2)
            errTest = 1 - accuracy(Ttest, Y)
            print('Epoch {}: Training loss = {:.2}, Training error = {:.2%}, Test error = {:.2%}'.format(k+1, lossTrain, errTrain, errTest))
            lossList.append(lossTrain)
            errTrainList.append(errTrain)
            errTestList.append(errTest)

    plt.figure()
    plt.plot(errTestList)
    plt.plot(errTrainList)
    plt.title('Figure 7, Question 4(d): training and test error for batch gradient descent')
    plt.xlabel('Training time')
    plt.ylabel('Error')

    plt.figure()
    plt.plot(lossList)
    plt.title('Figure 8, Question 4(d): mean training loss for batch gradient descent')
    plt.xlabel('Training time')
    plt.ylabel('Loss')

    if K > 5:
        plt.figure()
        plt.plot(errTestList[-100:])
        plt.plot(errTrainList[-100:])
        plt.title('Figure 9, Question 4(d): training and test error for last 500 epochs of bgd')
        plt.xlabel('Training time')
        plt.ylabel('Error')

        plt.figure()
        plt.plot(lossList[-100:])
        plt.title('Figure 10, Question 4(d)" mean training loss for last 500 epochs of bgd')
        plt.xlabel('Training Time')
        plt.ylabel('Loss')

# Q4 (e)

print('\n')
print ('Question 4(e).')
print('---------')
bgd(W[0], b[0], 0.1, 0.01, 1000)


# Q4 (g)

def sgd(W1, b1, lrate, alpha, sigma, K, batchSize, mom):
    _, M1 = np.shape(W1)
    _, M2 = np.shape(Ttrain)
    W2 = sigma * rnd.randn(M1, M2)
    b2 = np.zeros([M2])

    DW2sum = np.zeros([M1, M2])
    Db2sum = np.zeros([M2])
    lossList = []
    errTrainList = []
    errTestList = []
    minTestErr = np.inf

    for e in range (K):
        N1 = 0

        while N1 < Ntrain:
            N2 = np.min([N1 + batchSize, Ntrain])
            X = Xtrain[N1:N2]
            T = Ttrain[N1:N2]
            N1 = N2
            H, Y = predict(X, W1, W2, b1, b2)
            DW2, Db2 = gradient(H, Y, T)
            DW2 = DW2 + alpha * W2
            DW2sum = mom * DW2 + lrate * DW2
            Db2sum = mom * Db2 + lrate * Db2 
            W2 = W2 - DW2sum
            b2 = b2 - Db2sum

        if np.mod(e, 5) == 0:
            H, Ytrain = predict(Xtrain, W1, W2, b1, b2)
            lossTrain = loss(Ttrain, Ytrain)
            errTrain = 1 - accuracy(Ttrain, Ytrain)
            H, Ytest = predict(Xtest, W1, W2, b1, b2)
            errTest = 1 - accuracy(Ttest, Ytest)
            print('Epoch {}: Training loss = {:.2}, Training error = {:.2%}, Test error = {:.2%}'.format(e+1, lossTrain, errTrain, errTest))
            lossList.append(lossTrain)
            errTrainList.append(errTrain)
            errTestList.append(errTest)

            if errTest < minTestErr:
                minTestErr = errTest
    print('\nMinimum Test Error = {:.2%} \n'.format(minTestErr))
    plt.figure()
    plt.plot(errTestList)
    plt.plot(errTrainList)
    plt.title('Figure 11, Question 4(g): training and test error for stochastic gradient descent')
    plt.xlabel('Training time')
    plt.ylabel('Error')

    plt.figure()
    plt.plot(lossList)
    plt.title('Figure 12, Question 4(g): mean training loss for stochastic gradient descent')
    plt.xlabel('Training time')
    plt.ylabel('Loss')

# Q4 (h)

print('\n')
print('Question 4(h).')
print('---------')
mom = 0.8
batchSize = 100
print('batch size = {}, momentum = {}'.format(batchSize, mom))
print('')
sgd(W[0], b[0], 0.1, 0.0001, 0.01, 50, batchSize, mom)

