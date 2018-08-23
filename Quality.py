import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas
raw = []

with open('FGL.csv', 'rb') as csvfile:
    rawIn = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in rawIn:
        if row[0] == "Name": #Skip header row
            continue
        curRow = []
        for value in row[4:]:   #Remove name,IP,G,Pitches for data
            if(value == ''):
                curRow.append(0)
            else:
                curRow.append(float(value))
        raw.append(curRow)      
data = np.array(raw)
np.random.shuffle(data)
X = data[:,0:-1] #Data is everything except last column
Y = data[:,-1:] #Values are last column, Pitches/IP
print(X[0:3])
print(Y[0:3])
#1378 values in training set, 1100-278 split
X_train = X[0:1100]
Y_train = Y[0:1100]
X_test = X[1100:]
Y_test = Y[1100:]

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#10 Input values, 1 Prediction
X = tf.placeholder(tf.float32, shape=(None, 2))
Y = tf.placeholder(tf.float32, shape=(None, 1))
keep_prob = tf.placeholder(tf.float32)

#Set up weight and bias matrix of variables
W1 = tf.get_variable("W1", [2, 10], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", [10], initializer=tf.zeros_initializer())
W2 = tf.get_variable("W2", [10,10], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", [10], initializer=tf.zeros_initializer())
W3 = tf.get_variable("W3", [10,5], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", [5], initializer=tf.zeros_initializer())
W4 = tf.get_variable("W4", [5,1], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", [1], initializer = tf.zeros_initializer())

#Link layers of NN
X = tf.nn.l2_normalize(X)
Z1 = tf.matmul(X, W1)+b1 
A1 = tf.nn.relu(Z1)
Z2 = tf.matmul(A1, W2)+b2
A2 = tf.nn.relu(Z2)
Z3 = tf.matmul(A2, W3)+b3
A3 = tf.nn.relu(Z3)
Z4 = tf.matmul(A3, W4)+b4
A4 = tf.nn.relu(Z3)
Z4 = tf.matmul(A4, W4)+b4

def compute_cost(Z4, Y):
    cost = tf.reduce_mean((Z4-Y)**2)
    return cost

cost = compute_cost(Z4, Y)
starter_learning_rate = 0.05
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100, 0.85, staircase=True)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
init=tf.global_variables_initializer()
outData = Z4

batch_size = 100
with tf.Session() as sess:
    sess.run(init)
    train_costs = []
    test_costs = []
    for epoch in range(110):
        for i in range(0, 1100, batch_size):
            sess.run(optimizer, feed_dict={X:X_train[i:i+batch_size], Y:Y_train[i: i+batch_size], keep_prob : 0.65})
        train_costs.append(sess.run(cost, feed_dict={X:X_train, Y:Y_train, keep_prob : 1}))
        test_costs.append(sess.run(cost, feed_dict={X:X_test, Y:Y_test, keep_prob : 1}))
        if epoch%10 == 9:
            print("Test costs after " + str(epoch+1)+ " epochs: " + str(train_costs[-1]))
    iterations = list(range(110))
    plt.plot(iterations, train_costs, label="Train")
    plt.plot(iterations, test_costs, label ='Test')
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.show

    accuracy = tf.reduce_mean((Z4-Y)**2)
    train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob : 1})
    test_accuracy = accuracy.eval({X: X_test, Y: Y_test, keep_prob : 1})
    print("RMSE Train:", np.sqrt(train_accuracy))
    print("RMSE Test:", np.sqrt(test_accuracy))
        
    with open('current.csv', 'rb') as curCSV:
        curIn = csv.reader(curCSV, delimiter=',', quotechar='|')
        rawData = []    
        names = []
        for row in curIn:
            if(row[0] == 'Name'):
                continue      
            curRow = []
            names.append(row[0])
            for value in row[1:]:
                if(value == ''):
                    curRow.append(0)
                else:
                    curRow.append(float(value))     
            rawData.append(curRow)
    data = np.array(rawData)
    finalOutData = sess.run(outData, feed_dict={X:data})
    output = np.column_stack((np.asarray(names),finalOutData))
    df = pandas.DataFrame(output)    
    df.to_csv("estimted.csv")
