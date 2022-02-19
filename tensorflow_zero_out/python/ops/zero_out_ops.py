from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from tensorflow.python.ops import summary_ops_v2
import tensorflow as tf
from tensorflow.python.eager import context

import functools

counter_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_zero_out_ops.so'))
#create_counter = counter_ops.summary_writer

increment_counter = counter_ops.influx_writer

def summary_writer(ip, port, token, project, experiment):
    return summary_ops_v2._ResourceSummaryWriter(create_fn = lambda:
                                               counter_ops.influx_writer(shared_name=context.anonymous_name()),
                                               init_op_fn =
                                               functools.partial(
            counter_ops.create_testt_file_writer,
            url=ip,
            port=port,
            token=token,
            project = project,
            experiment= experiment,
            max_queue=tf.constant(10)))
