import tensorflow as tf
import tensorflow_zero_out


@tf.function
def test():
    writer = tensorflow_zero_out.python.ops.zero_out_ops.summary_writer()
    writer.set_as_default();
    #with writer.as_default():
    tf.summary.scalar("chummy", 0.1234, step=6)
    tf.summary.scalar("chummy1", 1.1234, step=16)
    tf.summary.histogram("dummy_{}".format(str("ee")),
                             tf.cast(tf.constant([1, 2, ]), tf.double), step=6)
    tf.summary.scalar("yummy", 99.1234, step=6)
    print("naaay")
test()
