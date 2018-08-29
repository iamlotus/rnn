import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

# import data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

# hypterparameter
lr=0.001
training_iters =100000
batch_size=128
n_inputs=28               # MNIST data input (img shape: 28*28)
n_steps=28                # time stamp
n_hidden_units=128        # neurons in hidden layer
n_classes=10              # MNIST classes(0-9 digits)

# x,y placeholder
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])

# weights and bias 
weights={
    # Shape (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    # Shape (128，10）
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

bias={
    # Shape(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units])),
    # Shape(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    # Input X is 3-D, we must transform it to 2-D to apply matrix multiple
    # X==>(128 batches * 28 step, 28 inputs)
    X=tf.reshape(X,[-1,n_inputs])
    X_in=tf.matmul(X,weights['in'])+biases['in']

    # X==>(128 batches, 28 steps, 28 inputs), transform back to 3-D tensor
    X_in=tf.reshape(X,[-1,n_steps,n_inputs])

    # Use basic RNN cell
    rnn_cell=tf.nn.rnn_cell.BasicRNNCell(n_hidden_units)

    # Init state is all zero
    init_state=rnn_cell.zero_state(batch_size,dtype=tf.float32)

    out_puts,final_state=tf.nn.dynamic_rnn(rnn_cell,X_in,initial_state=init_state,time_major=False)
    # Here we use final_state to calculate results( outputs[-1] is same with final_state in this example)
    results=tf.matmul(final_state,weights['out'])+bias['out']
    return results

predictions=RNN(x,weights,bias)
loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions,y))
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

correct_pred=tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=0
    while step * batch_size <training_iters:
        batch_xs,batch_ys=mnist.train.next_batch(batch_size)
        batch_xs =batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys
        })

        if step %20 ==0:
            print (sess.run(accuracy,feed_dict={
            x:batch_xs,
            y:batch_ys
        }))

        step+=1