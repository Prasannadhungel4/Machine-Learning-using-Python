import pandas as pd
import tensorflow as tf
597693

n_inputs = 9
n_hidden1 = 30
n_hidden2 = 15
n_outputs = 2

X = tf.placeholder(shape=(None, n_inputs), name="X")
y = tf.placeholder(shape=(None), name="y")

def 


