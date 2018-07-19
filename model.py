import tensorflow as tf
import time
import numpy as np

class model():
    def __init__(self, feature_size, h_size, class_size, dataset):
        self.feature_size = feature_size
        self.h_size = h_size
        self.class_size = class_size
        # self.input_X = input_X
        # self.input_Y = input_Y

        self.create_parameters()
        self.initialize_parameters()
        self.dataset = dataset


        self.input_X = self.dataset.train_idx_version
        self.input_Y = self.dataset.train_label

        self.input_testX = self.dataset.test_idx_version
        self.input_testY = self.dataset.test_label

    def create_parameters(self):
        self.X = tf.placeholder(tf.float32, shape=(self.feature_size, None))
        self.Y = tf.placeholder(tf.float32, shape=(self.class_size, None))
        # self.y_hat = tf.placeholder(tf.float32, shape=(self.class_size, None))

    def initialize_parameters(self):
        self.W1 = tf.get_variable("W1", [self.h_size, self.feature_size], initializer=tf.contrib.layers.xavier_initializer())
        self.W2 = tf.get_variable("W2", [self.class_size, self.h_size], initializer=tf.contrib.layers.xavier_initializer())
        self.b1 = tf.get_variable("b1", [self.h_size, 1], initializer=tf.zeros_initializer())
        self.b2 = tf.get_variable("b2", [self.class_size, 1], initializer=tf.zeros_initializer())
        self.parameters = {"W1": self.W1, "W2": self.W2, "b1": self.b1, "b2": self.b2}

    def forward(self):
        hidden_layer = tf.add(tf.matmul(self.parameters["W1"], self.X), self.parameters["b1"])
        y_hat = tf.nn.softmax(tf.add(tf.matmul(self.parameters["W2"], hidden_layer), self.parameters["b2"]), axis = 0, name="outputlayer")
        # accuracy = tf.metrics.accuracy(self.Y, y_hat)
        return y_hat
    def get_accuracy(self, y_hat):

        correct_prediction = tf.equal(tf.argmax(y_hat, 0), tf.argmax(self.Y, 0))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return accuracy, correct_prediction


    def compute_cost(self,y_hat):
        # cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=self.Y)
        cost = -tf.reduce_mean(tf.log(y_hat) * self.Y)
        return cost

    def read_data_batch(self, batch_size=8):
        batch_X, batch_Y = tf.train.batch([self.input_X, self.input_Y], batch_size=batch_size, num_threads=8)
        return batch_X, batch_Y

    def train(self, model_name, batch_size = 8, epochs = 5, learning_rate = 0.01, decay_rate = 0.9, global_step = 0.9):
        global_step = tf.Variable(0, trainable=False)
        init = tf.global_variables_initializer()
        init_op = tf.local_variables_initializer()
        y_hat = self.forward()
        cost = self.compute_cost(y_hat)

        learning_rate = tf.train.exponential_decay(learning_rate, global_step = global_step, decay_rate = decay_rate,
                                                   decay_steps=100000, staircase=True)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        saver = tf.train.Saver()
        acc, correct_prediction = self.get_accuracy(y_hat)
        # batch_X, batch_Y = self.read_data_batch(batch_size)
        with tf.Session() as sess:
            sess.run(init)
            sess.run(init_op)
            print("start!!")
            for epoch in range(epochs):
                start_time = time.time()
                # print(len(self.input_X)/batch_size)
                # print(self.input_X)
                for i in range(int(len(self.input_X)/batch_size)):
                # for i in range(10):
                    # print ("batch {}".format(i))
                    # x_, y_ = sess.run([batch_X, batch_Y])
                    # print(x_.shape)
                    # print(y_.shape)
                    if (i+1) * batch_size >= len(self.input_X):
                        x_ = self.dataset.todense(self.input_X[i * batch_size:]).T
                        y_ = self.input_Y.T[i * batch_size:].T
                    else:
                        x_ = self.dataset.todense(self.input_X[i * batch_size : (i+1) * batch_size]).T
                        y_ = self.input_Y.T[i * batch_size : (i+1) * batch_size].T
                    # np.savetxt("x_"+str(i)+".csv", x_, delimiter=",")
                    # np.savetxt("y_"+str(i)+".csv", y_, delimiter=",")
                    # print("x")
                    # print(x_)
                    # print(x_.shape)
                    # print("y")
                    # print(y_)
                    # print(y_.shape)
                    _, loss, prediction = sess.run([optimizer, cost, y_hat], feed_dict = {self.X: x_, self.Y: y_})

                testx_ = self.dataset.todense(self.dataset.test_idx_version).T
                testy_ = self.dataset.test_label
                # print("testy_")
                # print(testy_)
                # test_prediction = sess.run([self.y_hat], feed_dict={self.X: testx_})
                # test_prediction = np.asarray(test_prediction).reshape(np.asarray(test_prediction).shape[1], -1)
                # print(np.asarray(test_prediction).reshape(np.asarray(test_prediction).shape[1], -1), np.asarray(test_prediction).reshape(np.asarray(test_prediction).shape[1], -1).shape)
                # print(testy_, testy_.shape)
                # acc = self.get_score()

                score = sess.run([acc], feed_dict = {self.X: testx_, self.Y: testy_})
                tmp, tmpprediction = sess.run([correct_prediction, y_hat], feed_dict = {self.X: testx_, self.Y: testy_})
                np.savetxt("pred.csv", tmpprediction, delimiter=",")
                np.savetxt("obs.csv", testy_, delimiter=",")
                # print(tmpprediction)
                # print(tmp)
                # print(tmp.shape)
                # print(test_prediction)
                end_time = time.time()
                print("epoch {}th: Accuracy: {}, Time: {} sec, Loss: {}".format(epoch, score, end_time-start_time, loss))
            save_path = saver.save(sess, model_name)
            return loss



    def predict(self, model_name):
        tf.reset_default_graph()

        # saver = tf.train.import_meta_graph(model_name+'.meta')

        # y_hat = self.forward()
        # Later, launch the model, use the saver to restore variables from disk, and
        # do some work with the model.
        x_ = self.dataset.todense(self.input_testX).T
        y_ = self.input_testY

        print(x_.shape)
        print(y_.shape)
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_name+'.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))

            # op_to_restore =
            # graph = tf.get_default_graph()
            # w1 = graph.get_tensor_by_name("w1:0")
            # w2 = graph.get_tensor_by_name("w2:0")
            x_ = self.dataset.todense(self.input_testX).T
            y_ = self.input_testY

            print(x_.shape)
            print(y_.shape)
        # Restore variables from disk.
        # saver.restore(sess, save_model)
        # print("Model restored.")
        # # Check the values of the variables
        # print("v1 : %s" % v1.eval())
        # print("v2 : %s" % v2.eval())
            _, _, prediction = sess.run([y_hat], feed_dict={self.X: x_, self.Y: y_})
