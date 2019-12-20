""" TF model for point cloud autoencoder. PointNet encoder, FC decoder.
Using GPU Earth Mover's distance loss.

Author: Charles R. Qi
Date: May 2018
"""
import tensorflow as tf
import numpy as np
import math
import sys
import os
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# print(ROOT_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/nn_distance'))
from tf_ops.nn_distance import tf_nndistance
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/approxmatch'))
from  tf_ops.approxmatch import tf_approxmatch
from utils import tf_utils
from utils.tf_utils import _variable_on_cpu, _variable_with_weight_decay, batch_norm_for_fc, fully_connected

#  _variable_on_cpu _variable_with_weight_decay, batch_norm_for_fc


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl


def get_model(basisPsetFeat, is_training, bn_decay=None, num_points = 1024, batch_size = 10):
    """ Autoencoder for point clouds.
    Input:
        basisPsetFeat: TF tensor BxF
        is_training: boolean
        bn_decay: float between 0 and 1
    Output:
        net: TF tensor BxF, reconstructed point clouds
        end_points: dict
    """
    # basisPsetFeat = tf.reshape(basisPsetFeat, [basisPsetFeat.get_shape()[0].value, basisPsetFeat.get_shape()[1].value])
    # batch_size = batch_size

    # batch_size = basisPsetFeat.get_shape()[0].value
    # num_point = basisPsetFeat.get_shape()[1].value
    # point_dim = basisPsetFeat.get_shape()[2].value
    num_features = basisPsetFeat.get_shape()[1].value
    end_points = {}
    input = basisPsetFeat

    print(batch_size)
    print(num_points)


    # FC Decoder
    net = fully_connected(input, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    # net = fully_connected(net, 256, bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    # net = fully_connected(net, 256, bn=True, is_training=is_training, scope='fc4', bn_decay=bn_decay)
    net = fully_connected(net, 1024, bn=True, is_training=is_training, scope='fc5', bn_decay=bn_decay)
    net = fully_connected(net, num_points*3, activation_fn=None, scope='fc6')
    net = tf.reshape(net, (batch_size, num_points, 3))
    end_points['embedding'] = net
    return net, end_points

def get_loss(pred, label, end_points):
    """ pred: BxNx3,
        label: BxNx3, """
    dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(pred, label)
    pc_loss = tf.reduce_mean(dists_forward+dists_backward)
    end_points['pcloss'] = pc_loss

    match = tf_approxmatch.approx_match(label, pred)
    loss = tf.reduce_mean(tf_approxmatch.match_cost(label, pred, match))
    tf.summary.scalar('loss', loss)
    return loss, end_points


if __name__=='__main__':
    with tf.Graph().as_default():
    
        inputs = tf.zeros((10,1024))
        outputs = get_model(inputs, tf.constant(True), num_points=1024)
        print( outputs)
        loss = get_loss(outputs[0], tf.zeros((10,1024,3)), outputs[1])
        print(loss)

    # with tf.Session() as sess:
    #     # inputs = tf.zeros((1,1024))
    #     inputs = tf.Variable(tf.random_normal([1, 1024], stddev=0.35),
    #                   name="weights")
    #     outputs = get_model(inputs, tf.constant(True), num_points=1024)

    #     r_outputs = tf.Variable(tf.random_normal([1, 1024, 3], stddev=0.35),
    #                   name="weights")

    #     init_op = tf.initialize_all_variables()

    #     print( outputs)
    #     loss = get_loss(outputs[0], r_outputs, outputs[1])
    #     print(loss)

    #     sess.run(init_op)
    #     sess.run(outputs)
    #     print(sess.run(loss))
    #     print(sess.run(r_outputs))
