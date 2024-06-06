import tensorflow as tf
import numpy as np

class Estimator:

    def __init__(self, classifier_state_length, is_target_dqn, var_scope_name, bias_average):

        self.classifier_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, classifier_state_length], name="X_classifier")
        self.action_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, 1], name="X_datapoint")

        with tf.variable_scope(var_scope_name):
            
            fc1 = tf.contrib.layers.fully_connected(
                inputs=self.classifier_placeholder, 
                num_outputs=10, 
                activation_fn=tf.nn.sigmoid,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name],
            )

            fc2concat = tf.concat([fc1, self.action_placeholder], 1)

            fc3 = tf.contrib.layers.fully_connected(
                inputs=fc2concat, 
                num_outputs=5, 
                activation_fn=tf.nn.sigmoid,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name]
            )

            self.predictions = tf.contrib.layers.fully_connected(
                inputs=fc3, 
                num_outputs=1, 
                biases_initializer=tf.constant_initializer(bias_average),
                activation_fn=None,
                trainable=not is_target_dqn,
                variables_collections=[var_scope_name],
            )

            tf.compat.v1.summary.histogram("estimator/q_values", self.predictions)
            self.summaries = tf.compat.v1.summary.merge_all()