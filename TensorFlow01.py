import tensorflow as tf
s=tf.InteractiveSession()

N=tf.placeholder("int64",name="Name")
result=tf.reduce_sum(tf.range(N)**2)

result.eval({N:10**8})

writer = tf.summary.FileWriter("path/to/log-directory", graph=s.graph)

