# coding: utf-8

import tensorflow as tf

def get_center_loss(features, labels,featuress, labelss ,alpha, num_classes):
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)

    centerss = tf.get_variable('centerss', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    labelss = tf.reshape(labelss, [-1])



##############################################################
    centers0=tf.unsorted_segment_mean(features,labels,num_classes)
    centers1=tf.unsorted_segment_mean(featuress,labelss,num_classes)
    EdgeWeights=tf.ones((num_classes,num_classes))-tf.eye(num_classes)
    margin=tf.constant(100,dtype="float32")
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    center_pairwise_dist = tf.transpose(norm(tf.expand_dims(centers1, 2) - tf.transpose(centers0)))
    loss_0= tf.reduce_sum(tf.multiply(tf.maximum(0.0, margin-tf.transpose(norm(tf.expand_dims(centers0, 2) - tf.transpose(centers0)))),EdgeWeights))
    # + tf.reduce_sum(tf.maximum(0.0, tf.pow((centers1 - centers0), 2))) \
    # + 0.01*tf.reduce_sum(tf.multiply(tf.maximum(0.0, tf.constant(200,dtype="float32")-center_pairwise_dist),EdgeWeights))

    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff

    # # 根据样本label,获取mini-batch中每一个样本对应的中心值
    # centers_batch1 = tf.gather(centerss, labelss)
    # # 当前mini-batch的特征值与它们对应的中心值之间的差
    # diff1 = centers_batch1 - featuress
    # unique_label1, unique_idx1, unique_count1 = tf.unique_with_counts(labelss)
    # appear_times1 = tf.gather(unique_count1, unique_idx1)
    # appear_times1 = tf.reshape(appear_times1, [-1, 1])
    # diff1 = diff1 / tf.cast((1 + appear_times1), tf.float32)
    # diff1 = alpha * diff1



    # 计算loss
    loss_1 = tf.nn.l2_loss(features - centers_batch)
    centers_update_op= tf.scatter_sub(centers, labels, diff)
    # centers_update_op1= tf.scatter_sub(centerss, labelss, diff1)


    return loss_0, loss_1, centers_update_op
