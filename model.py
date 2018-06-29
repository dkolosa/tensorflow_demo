import tensorflow as tf
import pandas as pd
from load_data import load_data

# load the data from dataset
x_scaled_training, y_scaled_training, x_scaled_testing, y_scaled_testing = load_data()

# model parameters
learning_rate = 0.001
training_epochs = 100
display_step = 5

# define inputs and outputs
number_of_inputs = 9
number_of_outputs = 1

# define how many nodes per layer
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50

# Define the layers of the neural network

# input layer
with tf.variable_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=(None, number_of_inputs))

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name="weight_1", shape=(number_of_inputs, layer_1_nodes), 
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(name='bias1', shape=([layer_1_nodes]), initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(x,weights) + bias)

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weight_2", shape=(layer_1_nodes, layer_2_nodes), 
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(name='bias2', shape=([layer_2_nodes]), initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output,weights) + bias)

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weight_3", shape=(layer_2_nodes, layer_3_nodes), 
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(name='bias2', shape=([layer_3_nodes]), initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output,weights) + bias)

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weight_4", shape=(layer_3_nodes, number_of_outputs), 
                              initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable(name='bias4', shape=([number_of_outputs]), initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output,weights) + bias)

with tf.variable_scope('cost'):
    Y = tf.placeholder(dtype=tf.float32, shape=(None,1))
    cost = tf.reduce_mean(tf.squared_difference(prediction,Y))

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    training_writer = tf.summary.FileWriter('./log/training', session.graph)
    testing_writer = tf.summary.FileWriter('./log/testing', session.graph)

    for epoch in range(training_epochs):

        session.run(optimizer, feed_dict={x: x_scaled_training, Y: y_scaled_training})
        # print("Training pass: {}".format(epoch))

        if epoch % display_step == 0:
            training_cost, training_summary = session.run([cost, summary], feed_dict={x: x_scaled_training, Y: y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary], feed_dict={x: x_scaled_testing, Y: y_scaled_testing})

            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)
            # print(epoch, training_cost, testing_cost)

    print('Training is Complete!!')

    final_training_cost = session.run(cost, feed_dict={x: x_scaled_training, Y: y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={x: x_scaled_testing, Y: y_scaled_testing})

    print("final training cost: {}".format(final_training_cost))
    print("final testing cost: {}".format(final_testing_cost))

    save_path = saver.save(session, "log/trained_model.ckpt")
    


