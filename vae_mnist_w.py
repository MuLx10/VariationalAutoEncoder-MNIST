import numpy as np #matrix math
import tensorflow as tf #machine learning
import matplotlib.pyplot as plt #plotting


# Import MINST data
#The MNIST data is split into three parts: 55,000 data points of training data 
#10,000 points of test data and 5,000 points of validation data 
#very MNIST data point has two parts: an image of a handwritten digit 
#and a corresponding label. 
#We'll call the images "x" and the labels "y". 
#Both the training set and test set contain images and their corresponding labels; 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)

n_pixels = 28*28

X = tf.placeholder(tf.float32, shape=([None, n_pixels]))

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.1)

    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def FC_layer(X, W, b):
    # print b.get_shape()
    return tf.matmul(X, W) + b



latent_dim = 20
#num neurons
h_dim = 500

#layer 1
W_enc = weight_variable([n_pixels, h_dim], 'W_enc')
b_enc = bias_variable([h_dim], 'b_enc')
# tanh activation function to replicate original model
#The tanh function, a.k.a. hyperbolic tangent function, 
#is a rescaling of the logistic sigmoid, such that its outputs range from -1 to 1.
#tanh or sigmoid? Whatever avoids the vanishing gradient problem!
h_enc = tf.nn.tanh(FC_layer(X, W_enc, b_enc))

#layer 2
W_mu = weight_variable([h_dim, latent_dim], 'W_mu')
b_mu = bias_variable([latent_dim], 'b_mu')
mu = FC_layer(h_enc, W_mu, b_mu) #mean

#instead of the encoder generating a vector of real values, 
#it will generate a vector of means and a vector of standard deviations.
#for reparamterization
W_logstd = weight_variable([h_dim, latent_dim], 'W_logstd')
b_logstd = bias_variable([latent_dim], 'b_logstd')
logstd = FC_layer(h_enc, W_logstd, b_logstd)

# reparameterization trick - lets us backpropagate successfully
#since normally gradient descent expects deterministic nodes
#and we have stochastic nodes
#distribution
noise = tf.random_normal([1, latent_dim])
#sample from the standard deviations (tf.exp computes exponential of x element-wise) 
#and add the mean 
#this is our latent variable we will pass to the decoder
print noise.get_shape(),logstd.get_shape()
z = mu + tf.multiply(noise, tf.exp(.5*logstd))

W_dec = weight_variable([latent_dim, h_dim], 'W_dec')
b_dec = bias_variable([h_dim], 'b_dec')
#pass in z here (and the weights and biases we just defined)
h_dec = tf.nn.tanh(FC_layer(z, W_dec, b_dec))


#layer 2, using the original n pixels here since thats the dimensiaonlty
#we want to restore our data to
W_reconstruct = weight_variable([h_dim, n_pixels], 'W_reconstruct')
b_reconstruct = bias_variable([n_pixels], 'b_reconstruct')
#784 bernoulli parameters output
reconstruction = tf.nn.sigmoid(FC_layer(h_dec, W_reconstruct, b_reconstruct))


log_likelihood = tf.reduce_sum(X*tf.log(reconstruction + 1e-9)+(1 - X)*tf.log(1 - reconstruction + 1e-9), reduction_indices=1)

KL_term = -.5*tf.reduce_sum(1 + 2*logstd - tf.pow(mu,2) - tf.exp(2*logstd), reduction_indices=1)

# This allows us to use stochastic gradient descent with respect to the variational parameters
variational_lower_bound = tf.reduce_mean(log_likelihood - KL_term)
optimizer = tf.train.AdadeltaOptimizer().minimize(-variational_lower_bound)



init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
## Add ops to save and restore all the variables.
saver = tf.train.Saver()


if __name__ == '__main__':

    import time #lets clock training time..

    num_iterations = 1000000
    recording_interval = 1000
    #store value for these 3 terms so we can plot them later
    variational_lower_bound_array = []
    log_likelihood_array = []
    KL_term_array = []
    iteration_array = [i*recording_interval for i in range(num_iterations/recording_interval)]
    for i in range(num_iterations):
        # np.round to make MNIST binary
        #get first batch (200 digits)
        x_batch = np.round(mnist.train.next_batch(200)[0])
        #run our optimizer on our data
        sess.run(optimizer, feed_dict={X: x_batch})
        if (i%recording_interval == 0):
            #every 1K iterations record these values
            vlb_eval = variational_lower_bound.eval(feed_dict={X: x_batch})
            print "Iteration: {}, Loss: {}".format(i, vlb_eval)
            variational_lower_bound_array.append(vlb_eval)
            log_likelihood_array.append(np.mean(log_likelihood.eval(feed_dict={X: x_batch})))
            KL_term_array.append(np.mean(KL_term.eval(feed_dict={X: x_batch})))



    # saver.save(sess)
    save_path = saver.save(sess, "stored_model/Trained Bernoulli VAE2.data")
    print("Model saved in file: %s" % save_path)

    plt.figure()
    #for the number of iterations we had 
    #plot these 3 terms
    plt.plot(iteration_array, variational_lower_bound_array)
    plt.plot(iteration_array, KL_term_array)
    plt.plot(iteration_array, log_likelihood_array)
    plt.legend(['Variational Lower Bound', 'KL divergence', 'Log Likelihood'], bbox_to_anchor=(1.05, 1), loc=2)
    plt.title('Loss per iteration')

else:
    import os
    import cv2
    load_model = True
    if load_model:
        saver.restore(sess, os.path.join(os.getcwd(), "stored_model/Trained Bernoulli VAE"))

    num_pairs = 10
    image_indices = np.random.randint(0, 200, num_pairs)
    #Lets plot 10 digits
    print("model loaded")
    for pair in range(num_pairs):
        print(pair,num_pairs)
        #reshaping to show original test image
        x = np.reshape(mnist.test.images[image_indices[pair]], (1,n_pixels))
        plt.figure()
        x_image = np.reshape(x, (28,28))
        plt.subplot(121)
        plt.imshow(x_image)
        #reconstructed image, feed the test image to the decoder
        x_reconstruction = reconstruction.eval(feed_dict={X: x})
        #reshape it to 28x28 pixels
        x_reconstruction_image = (np.reshape(x_reconstruction, (28,28)))
        #plot it!
        plt.subplot(122)
        plt.imshow(x_reconstruction_image)

        cv2.imshow('0',x_image)
        cv2.waitKey(100)
        cv2.imshow('1',x_reconstruction_image)
        cv2.waitKey(2000)

        # cv2.imwrite('img/'+str(pair)+'-0.jpg',x_image)
        # cv2.imwrite('img/'+str(pair)+'-1.jpg',x_reconstruction_image)

    cv2.destroyAllWindows()