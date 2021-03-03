import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# break the book into input and target batches with integers representing each character
def generate_batches(text, batch_size, sequence_length, dictionary):
    block_length = len(text) // batch_size

    # initialize input batches
    X_batches = []
    # initialize target batches
    Y_batches = []

    # iterate over all blocks by step size sequence length
    for i in range(0, block_length, sequence_length):
        X_batch = []
        Y_batch = []

        # iterate over all batches as columns
        for j in range(batch_size):
            # initialize input and target sequences
            X_sequence = np.zeros(sequence_length-1)
            Y_sequence = np.zeros(sequence_length-1)
            # define start and end point of each sequence
            start = j * block_length + i
            end = min(start + sequence_length, j * block_length + block_length)

            # iterate over columns within a batch
            for k in range(end - start - 1):
                # replace each character by its representing integer and place it into the sequence
                X_sequence[k] = dictionary[text[start + k]]
                Y_sequence[k] = dictionary[text[start + k + 1]]

            # append all sequences of a batch
            X_batch.append(X_sequence)
            Y_batch.append(Y_sequence)

        # append all batches to an array of batches
        X_batches.append(np.array(X_batch, dtype=int))
        Y_batches.append(np.array(Y_batch, dtype=int))

    return X_batches, Y_batches

def main():
    seed = 0
    tf.reset_default_graph()
    tf.set_random_seed(seed=seed)

    # Read file and store it in book variable
    book = open('MonteCristo.txt', 'r').read()
    # Make all characters lower case
    book = book.lower()
    # replace all new lines with empty strings
    book = book.replace('\n', ' ')\

    # putting the data in a pandas dataframe, each character seperately
    char_df = pd.DataFrame({'chars': list(book)})

    # total unique characters
    sum = char_df.count()

    # compute frequency of unique characters and add it to the data frame
    char_df = char_df['chars'].value_counts().to_frame('frequency')
    char_df['frequency'] = char_df['frequency']/char_df['frequency'].sum()
    # represent characters with index value based on frequency
    char_df = char_df.reset_index()

    # write data frame to file
    f = open('book_dataframe', 'w+')
    f.write(str(char_df))
    f.close()

    # plot the frequency distribution of each unique character
    plt.pie(char_df['frequency'], labels=char_df['index'], autopct='%1.1f%%', startangle=90)
    plt.show()

    # storing characters and replacement values, each pair a separate row
    replacements = []
    for i in range(len(char_df)):
        replacements = np.append(replacements, [char_df.loc[i, 'index'], i])
        char_df.loc[i, 'index'] = i
    replacements = np.reshape(replacements, (-1, 2))
    # turn replacement array into a dictionary with key: character and value: replacement integer
    dictionary = dict(replacements)

    # create the input X_batches and target Y_batches
    subsequence_size = 256
    batch_size = 16
    X_batches, Y_batches = generate_batches(book, batch_size, subsequence_size, dictionary)
    batches_amount = len(Y_batches)

    # Model parameters
    hidden_units = 256  # Number of recurrent units
    # Training procedure parameters
    learning_rate = 1e-2
    n_epochs = 5
    k = len(char_df)

    # Model definition
    X_int = tf.placeholder(shape=[None, None], dtype=tf.int64)
    Y_int = tf.placeholder(shape=[None, None], dtype=tf.int64)

    # One-hot encoding X_int
    X = tf.one_hot(X_int, depth=k)  # shape: (batch_size, sequence_length, k)
    # One-hot encoding Y_int
    Y = tf.one_hot(Y_int, depth=k)  # shape: (batch_size, sequence_length, k)

    # LSTM cell
    cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
    # Initialize at zero state
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # rnn_outputs shape: (batch_size, sequence_length, hidden_units)
    rnn_outputs, \
        final_state = tf.nn.dynamic_rnn(cell, X,
                                        initial_state=init_state)

    # rnn_outputs_flat shape: ((batch_size * sequence_length), hidden_units)
    rnn_outputs_flat = tf.reshape(rnn_outputs, [-1, hidden_units])

    # Weights and biases for the output layer
    Wout = tf.Variable(tf.truncated_normal(shape=(hidden_units, k), stddev=0.1))
    bout = tf.Variable(tf.zeros(shape=[k]))

    # Z shape: ((batch_size * sequence_length), 2)
    Z = tf.matmul(rnn_outputs_flat, Wout) + bout
    # Normalize to probability vector
    Z_normalized = tf.nn.softmax(Z)

    Y_flat = tf.reshape(Y, [-1, k]) # shape: ((batch_size * max_len), 2)

    # Loss definition
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_flat, logits=Z)
    loss = tf.reduce_mean(loss)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    session = tf.Session()
    session.run(tf.global_variables_initializer())

    # initialize start state as initial zero state
    current_state = session.run(init_state)

    losses = []
    c = 100
    # run over all epochs
    for e in range(1, n_epochs + 1):
        # feed batch after batch
        # each time updating the current state and feed it as the next init_state
        for b in range(batches_amount - 1):
            feed = {X_int: X_batches[b], Y_int: Y_batches[b], init_state: current_state}
            l, _, current_state = session.run([loss, train, final_state], feed)
            # store the evolution of the loss value
            losses.append(l)
            # each 100th iteration pring the loss value
            if b % c == 0:
                print('Epoch: {0}. Batch: {1}. Loss: {2}.'.format(e, b + 1, l))

    # plot the loss evolution
    plt.plot(losses)
    plt.gca().legend(('training loss'))
    plt.show()

    # initialize the output array
    output = np.zeros((1, 256), dtype=int)
    # run 2 times to get 20 output sequences
    for i in range(2):
        # initialize the local output sequence
        output_loc = np.zeros((16, 256), dtype=int)
        # run over all 16 sequences
        for i in range(len(output_loc)):
            # initialize each sequence with a different starting value, starting from 1
            output_loc[i,0] = i+1
         # take the startinng value and run over all other values in the sequence
        for j in range(len(output_loc[1])-1):
            # predict the next value given the current value, column by column for all 16 sequences
            feed = {X_int: np.reshape(output_loc[:, j], (-1, 1)), init_state: current_state}
            # store the predictions and the current state
            # always feed back the current state as the init state
            predictions, current_state = session.run([Z_normalized, final_state], feed)
            # run over each sequence seperately
            for i in range(len(output_loc)):
                # probabilistically predict the next value according to the value's prediction vector
                index = 0
                r = np.random.rand()
                while (r >= 0 and index < len(predictions[i])):
                    r -= predictions[i][index]
                    index += 1
                output_loc[i, j+1] = index-1
        # concatenate the two outputs to get a total of 32 sequences
        output = np.concatenate((output, output_loc))

    # inverse the dictionary key: representing integers, value: character
    inv_dictionary = {v: k for k, v in dictionary.items()}
    f = open('predictions_50k', 'w+')
    # run over each value in the output sequences and translate it back to the character
    # then append the characters back to a string
    for i in range(1, 21):
        output_txt = ""
        for j in range(len(output[0])):
            output_txt = output_txt + inv_dictionary[str(output[i][j])]
        f.write("{}) ".format(i) + output_txt)
        f.write('\n')

    f.close()

    session.close()

if __name__ == "__main__":
        main()