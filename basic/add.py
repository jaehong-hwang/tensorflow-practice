import tensorflow as tf

print "=== constant ==="

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print "node1:", node1, "node2:", node2
print "node3:", node3

sess = tf.Session()

print "sess.run(node1, node2):", sess.run([node1, node2])
print "sess.run(node3):", sess.run(node3)

print "\n=== placeholder ==="

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)
node3 = node1 + node2

print "node1:", node1, "node2:", node2
print "node3:", node3

print sess.run(node3, feed_dict={node1: 1, node2: 3.3})
print sess.run(node3, feed_dict={node1: [1,3], node2: [2,4]})
