import tensorflow as tf
from tqdm import tqdm
from midi_utils import *
import numpy as np
from six.moves import cPickle as pickle
from music21 import *
from glob import glob
import os


### RNN-RBM hyperparameters ###

timesteps = 5
# RBM visible layer size. At each timestep, the input vector will be of size 2*span
# The input vector is of 1s and 0s. The first `span` values
# encode note-on events, or when a chord starts, and the remaining `span` values
# encode note-off events. Refer to this http://danshiebler.com/img/Matrix.png
n_visible = span * 2
n_visible = n_visible * timesteps
# eg. [[1 0],[0 0],[0 1]] num_timesteps=3, span=1,
# here the note event starts at timestep t=1 and finishes at timestep t=3

# RBM hidden layer size
n_hidden = 50

# RNN hidden layer size
n_hidden_recurrent = 100

# Learning rate. Not tf.constant(...) because we change it during training
lr = tf.placeholder(tf.float32)

# NOTE: There's `batch_size` and `batch_size_`, and `lr` and `lr_`
# `batch_size` is a Tensor and `batch_size_` is a python value
batch_size_ = 100
lr_ = 0.01

# NOTE: In RNNs, only the hidden layers are recurrent layers

# Naming convention for parameters taken from here http://deeplearning.net/tutorial/rnnrbm.html
# u:RNN hidden, h:RBM hidden, v:RBM visible
# W:weights, b:bias
# x:input vector, u0:initial RNN state

# Refer to this image for the RNN-RBM architecture http://deeplearning.net/tutorial/_images/rnnrbm.png

# Placeholder for input vector `x`
x  = tf.placeholder(tf.float32, [None, n_visible], name="x")
# Weights for the RBM
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
# Weights from RNN hidden layer at timestep t-1 to RBM hidden layer at timestep t
Wuh = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden], 0.0001), name="Wuh")
# Bias from RNN hidden layer at timestep t-1 to RBM hidden layer at timestep t
bh  = tf.Variable(tf.zeros([1, n_hidden], tf.float32), name="bh")
# Weights from RNN hidden layer at timestep t-1 to RBM visible layer at timestep t
Wuv = tf.Variable(tf.random_normal([n_hidden_recurrent, n_visible], 0.0001), name="Wuv")
# Bias from RNN hidden layer at timestep t-1 to RBM visible layer at timestep t
bv  = tf.Variable(tf.zeros([1, n_visible], tf.float32), name="bv")
# Weights from RBM visible layer at timestep t to RNN hidden layer at timestep t
Wvu = tf.Variable(tf.random_normal([n_visible, n_hidden_recurrent], 0.0001), name="Wvu")
# Weights from RNN hidden layer at timestep t-1 to RNN hidden layer at timestep t
Wuu = tf.Variable(tf.random_normal([n_hidden_recurrent, n_hidden_recurrent], 0.0001), name="Wuu")
# Bias from RNN hidden layer at timestep t-1 to RNN hidden layer at timestep t
bu  = tf.Variable(tf.zeros([1, n_hidden_recurrent],  tf.float32), name="bu")

# Initial RNN state
u0  = tf.Variable(tf.zeros([1, n_hidden_recurrent], tf.float32), name="u0")


# Bias from visible layer to hidden layer for RBM at timestep t
bv_t = tf.Variable(tf.ones([batch_size_, n_visible],  tf.float32), name="bv_t")
# Bias from hidden layer to visible layer for RBM at timestep t
bh_t = tf.Variable(tf.ones([batch_size_, n_hidden],  tf.float32), name="bh_t")


batch_size = tf.shape(x)[0]

tf.assign(bh_t, tf.tile(bh_t, [batch_size, 1]))
tf.assign(bv_t, tf.tile(bv_t, [batch_size, 1]))

# These functions are used within tf.Scan(...)
RNN_forw = lambda prev_t, sl: (tf.tanh(bu + tf.matmul(tf.reshape(sl, [1, n_visible]), Wvu) + tf.matmul(prev_t, Wuu)))
RNN_to_RBM_hidd = lambda _, prev_t: tf.add(bh, tf.matmul(prev_t, Wuh))
RNN_to_RBM_vis = lambda _, prev_t: tf.add(bv, tf.matmul(prev_t, Wuv))

u_t = tf.scan(RNN_forw, x, initializer=u0)

bv_t = tf.reshape(tf.scan(RNN_to_RBM_vis, u_t, tf.zeros([1, n_visible], tf.float32)), [batch_size, n_visible])
bh_t = tf.reshape(tf.scan(RNN_to_RBM_hidd, u_t, tf.zeros([1, n_hidden], tf.float32)), [batch_size, n_hidden])

epochs = 100



def preprocess_score(s, instr=None):
    """Preprocess a score `s` to facilitate training by:
        - Transposing each key to C
        - Change time signature
        - Quantize the score
        - Ignore notes faster than sixteenth notes
        - Using exclusively parts with `instr` as instrument
    :param s: a music21.stream.Score score
    :param instr: music21.instrument.Instrument to predict notes for
                Every instrument in a score has a different playing style
    """

    s = s.flat.notes


    # Ignore notes faster than a sixteenth note
    for n in s.recurse(skipSelf=True):
        if n.duration.quarterLength < 0.25:
            s.remove(n)


    # Transpose to C
    try:
        k = s.analyze('key')
    # Key for score `s` can't be found
    except analysis.discrete.DiscreteAnalysisException as e:
        print(e)
        return

    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    s = s.transpose(i)

    # Snap notes to sixteenth and eighth triplets, whichever is closer
    # (ie make sure each note has perfect timing)
    s = s.quantize()

    return s


def initialise_variables(songs, sess, weights_path="checkpoints/init.ckpt"):
    """Initialise and train a single RBM, and save the trained parameters. Those parameters are then
    used to initialise all RBMs in the RNN-RBM
    :param songs: All songs in the dataset
    :param sess: The current TensorFlow session
    :param weights_path: A path to file to load already trained parameters, if such file exists
    """
    saver = tf.train.Saver([W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0])

    if weights_path:
        saver.restore(sess, weights_path)
    else:
        # Train a simple RBM and use trained bv and bh initialise the bv_t and bh_t of each RBM in our RNN-RBM
        sess.run(tf.global_variables_initializer())
        epochs = 10

        print "Training weights to be used to initialise all RBMs"
        for epoch in tqdm(range(epochs)):
            for song in songs:
                sess.run(contrastive_divergence(k=1), feed_dict={x: song})
        saver.save(sess, weights_path)
    return sess


sample = lambda prob_dist: tf.floor(prob_dist + tf.random_uniform(tf.shape(prob_dist), 0, 1))



def gibbs_sample(x, W, bv, bh, k):
    def gibbs_step(i, k, xk):
        """Perform a single gibbs step
        :param i: Current loop iteration
        :param k: Number of gibbs steps to perform
        :param xk: The `x` sampled
        :return: Sample x after k steps, or `xk`
        NOTE: `i` and `k` are used by tf.while_loop(...)
        """
        v = xk  # Set the visible layer to be `xk`
        hk = sample(tf.sigmoid(tf.matmul(v, W) + bh))  # Forward propage to get the sampled hidden layer `hk`
        xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # Back propagate to get to resample `x`
        return i+1, k, xk

    [_, _, x_sample] = tf.while_loop(lambda i, n, *args: i < n, gibbs_step, [0, k, x],
                                     parallel_iterations=1, back_prop=False)
    return tf.stop_gradient(x_sample)


def free_energy_cost(x, W, bv, bh, k):
    """Get the free energy cost of the RBM. We can pass this cost directly into TensorFlow's optimizers.
    Used during training
    :param k: Number of steps
    """
    x_sample = gibbs_sample(x, W, bv, bh, k)

    # Free energy for visible layer configuration `v`
    free_energy = lambda v: -tf.reduce_sum(tf.log(1 + tf.exp(tf.matmul(v, W) + bh)), 1) - tf.matmul(v, tf.transpose(bv))

    # The cost is the difference in free energy between `x` and `x_sample`
    cost = tf.reduce_mean(tf.subtract(free_energy(x), free_energy(x_sample)))
    return cost


def contrastive_divergence(k, lr=.01):
    """Run k steps of contrastive divergence (also called CD)
    :param k: Number of steps
    :param lr: learning rate for CD
    :return: The update rules for CD
    """

    # Sample `x` and `h` from the probability distribution defined by the RBM

    # Sample visible layer `x`
    x_sample = gibbs_sample(x, W, bv, bh, k)
    # Sample hidden nodes `h`, starting from visible layer `x`
    h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
    # Sample hidden nodes `h`, starting from visible layer `x_sample`
    h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

    # Update W, bh, and bv, using the difference between `x` and `x_sample, and `h` and `h_sample`
    # We want those pairs of values to be as close as possible. This would mean that the probability
    # distribution modeled by the RBM is close the actual probability distribution in the dataset
    lr = tf.constant(lr, tf.float32)
    batch_size = tf.cast(tf.shape(x)[0], tf.float32)
    dW = tf.multiply(lr/batch_size, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
    dbv = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
    dbh = tf.multiply(lr/batch_size, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))

    return [W.assign_add(dW), bv.assign_add(dbv), bh.assign_add(dbh)]


def compose(n_timesteps, prime_timesteps=80):
    """Compose music using the trained RNN-RBM configuration
    :param n_timesteps: Number of timesteps to compose for
    :param prime_timesteps: Number of timesteps to go into the primer song before composing
    :return: A tf.Tensor of compose music
    """
    def compose_(i, k, prev_t, primer, x, pred):
        """Performs gibbs steps for each RBM in the RNN-RBM and generates a prediction
        :param i: Current loop iteration
        :param k: Number of iterations to perform
        :param prev_t: RNN hidden layer at timestep t-1
        :param primer: Song used to prime the RNN-RBM
        :param x: Input vector
        :param pred: Song being currently composed by the RNN-RBM
        NOTE: To be used within tf.while_loop(...)
        """

        bv_t = tf.add(bv, tf.matmul(prev_t, Wuv))
        bh_t = tf.add(bh, tf.matmul(prev_t, Wuh))

        x_out = gibbs_sample(primer, W, bv_t, bh_t, k=25)

        # Propagate through the RNN using the current output `x_out` and the RNN hidden layer at t-1, `prev_t`
        u_t  = (tf.tanh(bu + tf.matmul(x_out, Wvu) + tf.matmul(prev_t, Wuu)))

        # Append `x_out` to the predicted song `pred`
        pred = tf.concat(values=[pred, x_out], axis=0)
        return i+1, k, u_t, x_out, x, pred

    # RNN hidden layers
    u = tf.scan(RNN_forw, x, initializer=u0)
    u = u[int(np.floor(prime_timesteps / timesteps)), :, :]

    pred = tf.zeros([1, n_visible], tf.float32)
    # NOTE: Set `shape_invariants` in tf.while_loop because `pred` shape changes as outputs are appended
    # and if not set as TensorShape([None, 780]) an exception is thrown
    ts = tf.TensorShape  # To quickly define a TensorShape
    compose_loop_out = tf.while_loop(lambda i, n, *args: i < n, compose_, [tf.constant(1), tf.constant(n_timesteps), u,
                                     tf.zeros([1, n_visible], tf.float32), x, tf.zeros([1, n_visible], tf.float32)],
                                     shape_invariants=[ts([]), ts([]), u.get_shape(), ts([1, n_visible]), x.get_shape(), ts([None, 780])])
    # Try using eager execution instead of tf.while_loop(...)

    pred = compose_loop_out[5]
    return pred


def reshape_with_timesteps(x):
    x = x[:int(np.floor(x.shape[0]/timesteps)*timesteps)]
    return np.reshape(x, [x.shape[0]/timesteps, x.shape[1]*timesteps])




def run(dataset_paths, weights_filepath=None):
    """
    :param dataset_paths: The path to all datasets used to train.
                          See datasets available in the `datasets/` folder
    :param weights_filepath: Path to weights saved from previous training session
    """

    print 'Loading datasets ...'
    songs = []
    for path in dataset_paths:
        dataset_songs = []  # Songs from the current dataset
        for f in tqdm(glob(path+'/*.mid')):
            try:
                song = reshape_with_timesteps(np.array(midiToNoteStateMatrix(f)))
                if np.array(song).shape[0] > 50/timesteps:
                    dataset_songs.append(song)
            except Exception as e:
                print e
        songs.extend(dataset_songs)
    saver = tf.train.Saver([W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0])

    # Load saved weights and compose music (without training)
    if weights_filepath:
        print 'Composing ...'

        from random import choice
        primer_path = choice(glob(dataset_paths+'*.mid'))
        primer = reshape_with_timesteps(np.array(midiToNoteStateMatrix(primer_path)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, weights_filepath)

            # Prime the RNN-RBM to compose a song
            out = sess.run(compose(250), feed_dict={x: primer})

            # Write the song to midi
            out = np.reshape(out, (out.shape[0]*timesteps, 2*span))
            noteStateMatrixToMidi(out, name='compositions/out_'+primer_path.split('/')[-1])



    # Train and save trained weights after training
    else:
        print 'Training ...'

        u_t = tf.scan(RNN_forw, x, initializer=u0)

        bh_t = tf.reshape(tf.scan(RNN_to_RBM_hidd, u_t, tf.zeros([1, n_hidden], tf.float32)), [batch_size, n_hidden])
        bv_t = tf.reshape(tf.scan(RNN_to_RBM_vis, u_t, tf.zeros([1, n_visible], tf.float32)), [batch_size, n_visible])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gradients = optimizer.compute_gradients(free_energy_cost(x, W, bv_t, bh_t, 15), [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0])

        # Clip gradients to avoid exploding gradients (a common problem with RNNs)
        gradients = [(tf.clip_by_value(grad, -5.0, 5.0), hyperpar) for grad, hyperpar in gradients]

        with tf.Session() as sess:
            sess = initialise_variables(songs, sess)

            for epoch in range(epochs):
                loss_epoch = 0  # Loss for the current epoch
                for song in tqdm(songs):
                    for i in range(1, len(song), batch_size_):
                        _, cost = sess.run([optimizer.apply_gradients(gradients), free_energy_cost(x, W, bv_t, bh_t, 15)],
                                           feed_dict={x: song[i:i + batch_size_], lr: lr_ if epoch <= 10 else lr_/(epoch-10)})
                        loss_epoch += abs(cost)
                print '\nloss', loss_epoch/len(songs), 'at epoch', epoch
                # Save the RNN-RBM state for the current epoch
                save_path = saver.save(sess, "checkpoints/{}_{}.ckpt".format(epoch, loss_epoch))


# TODO Recently extended from `run(...)` to allow preprocessing and pickling but yet to properly test
def run_with_preprocessing(dataset_paths, weights_filepath=None):
    """
    :param dataset_paths: The path to all datasets used to train.
                          See datasets available in the `datasets/` folder
    :param weights_filepath: Path to weights saved from previous training session
    """

    songs = []
    for path in dataset_paths:
        pickle_filename = 'datasets/{}.pickle'.format(path.split('/')[-1])

        if os.path.exists(pickle_filename):
            print "Loading {} from pickle ...".format(path)
            with open(pickle_filename, 'rb') as f:
                dataset_songs = pickle.load(f)
        else:
            print "Preprocessing {}".format(path)
            # Preprocess songs and save preprocessed songs to midi
            for midi_file in tqdm(glob(path+'/*.mid')):
                s = converter.parse(midi_file)
                prep_s = preprocess_score(s)
                print midi_file
                split_midifile = midi_file.split('/')
                prep_folder, prep_filename = 'preprocessed/'+split_midifile[-2], split_midifile[-1]
                prep_filepath = prep_folder+'/'+prep_filename

                if not os.path.exists(prep_folder):
                    os.makedirs(prep_folder)

                prep_s.write('midi', fp=prep_filepath)


            dataset_songs = []  # Songs from the current dataset

            print "Loading preprocessed songs from {}".format(path)
            # Load preprocessed scores and save to pickle
            for midi_file in tqdm(glob(path+'/*.mid')):
                try:
                    song = reshape_with_timesteps(np.array(midiToNoteStateMatrix(midi_file)))
                    if np.array(song).shape[0] > 50/timesteps:
                        dataset_songs.append(song)
                except Exception as e:
                    print e

            print "Saving {} to pickle".format(path)
            with open(pickle_filename, 'wb') as f:
                pickle.dump(songs, f)

        songs.extend(dataset_songs)



    saver = tf.train.Saver([W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0])

    # Load saved weights and compose music (without training)
    if weights_filepath:
        print 'Composing ...'

        from random import choice
        primer_path = choice(glob(dataset_paths+'*.mid'))
        primer = reshape_with_timesteps(np.array(midiToNoteStateMatrix(primer_path)))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, weights_filepath)

            # Prime the RNN-RBM to compose a song
            out = sess.run(compose(250), feed_dict={x: primer})

            # Write the song to midi
            out = np.reshape(out, (out.shape[0]*timesteps, 2*span))
            noteStateMatrixToMidi(out, name='compositions/out_'+primer_path.split('/')[-1])



    # Train and save trained weights after training
    else:
        print 'Training ...'

        u_t = tf.scan(RNN_forw, x, initializer=u0)

        bh_t = tf.reshape(tf.scan(RNN_to_RBM_hidd, u_t, tf.zeros([1, n_hidden], tf.float32)), [batch_size, n_hidden])
        bv_t = tf.reshape(tf.scan(RNN_to_RBM_vis, u_t, tf.zeros([1, n_visible], tf.float32)), [batch_size, n_visible])

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        gradients = optimizer.compute_gradients(free_energy_cost(x, W, bv_t, bh_t, 15), [W, Wuh, Wuv, Wvu, Wuu, bh, bv, bu, u0])

        # Clip gradients to avoid exploding gradients (a common problem with RNNs)
        gradients = [(tf.clip_by_value(grad, -5.0, 5.0), hyperpar) for grad, hyperpar in gradients]

        with tf.Session() as sess:
            sess = initialise_variables(songs, sess)

            for epoch in range(epochs):
                loss_epoch = 0  # Loss for the current epoch
                for song in tqdm(songs):
                    for i in range(1, len(song), batch_size_):
                        _, cost = sess.run([optimizer.apply_gradients(gradients), free_energy_cost(x, W, bv_t, bh_t, 15)],
                                           feed_dict={x: song[i:i + batch_size_], lr: lr_ if epoch <= 10 else lr_/(epoch-10)})
                        loss_epoch += abs(cost)
                print '\nloss', loss_epoch/len(songs), 'at epoch', epoch
                # Save the RNN-RBM state for the current epoch
                save_path = saver.save(sess, "checkpoints/{}_{}.ckpt".format(epoch, loss_epoch))




#run(['datasets/Piano-midi.de'])
run(['datasets/ff'])
