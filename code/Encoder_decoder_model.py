################################################## 1. Import modules ###################################################
import gensim
import numpy as np
import ANMT.code.helper as hp
from ANMT.code.utils import *
import tensorflow as tf
import os
import time
########################################################################################################################

class Encoder_decoder:

    def __init__(self,source_word2vec_path,target_word2vec_path,source_hidden_dim,target_hidden_dim):

        # word properties
        self.PAD = 0

        # source language
        self.source_word2vec_model = gensim.models.Word2Vec.load(source_word2vec_path)
        self.source_vocab_size = len(self.source_word2vec_model.wv.index2word) + 1
        self.source_word_vec_dim = self.source_word2vec_model.vector_size
        self.source_lookup =[[0.] *self.source_word_vec_dim] + [x for x in self.source_word2vec_model.wv.syn0]
        self.source_EOS = self.source_word2vec_model.wv.vocab['EOS'].index + 1

        # target language
        self.target_word2vec_model = gensim.models.Word2Vec.load(target_word2vec_path)
        self.target_vocab_size = len(self.target_word2vec_model.wv.index2word) + 1
        self.target_word_vec_dim = self.target_word2vec_model.vector_size
        self.target_lookup = [[0.] *self.target_word_vec_dim] +[x for x in self.target_word2vec_model.wv.syn0]
        self.target_EOS = self.target_word2vec_model.wv.vocab['EOS'].index + 1

        # model hyper-parameters
        self.input_embedding_size = self.source_word_vec_dim
        self.encoder_hidden_units = source_hidden_dim
        self.decoder_hidden_units = target_hidden_dim

    def place_holders(self):
        # input, target shape : [max_time, batch_size]
        # sequence length shape : [batch_size]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_inputs')
        self.encoder_sequence_len = tf.placeholder(shape=(None), dtype=tf.int32, name='encoder_sequence_len')
        self.decoder_sequence_len = tf.placeholder(shape=(None), dtype=tf.int32, name='decoder_sequence_len')

        # Embedding matrix
        # denote lookup table as placeholder
        self.source_embeddings = tf.placeholder(shape=(self.source_vocab_size, self.input_embedding_size), dtype=tf.float32)
        self.target_embeddings = tf.placeholder(shape=(self.target_vocab_size, self.input_embedding_size), dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.source_embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.target_embeddings, self.decoder_inputs)

    # Encoder
    def encoder(self):
        self.encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)

        # only need final state encoder
        # semantic vector is final state of encoder
        _, self.encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell, self.encoder_inputs_embedded,sequence_length=self.encoder_sequence_len,
                                                                 dtype=tf.float32, time_major=True,scope="encoder",)

    # Decoder
    def decoder(self):
        self.decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)

        self.decoder_outputs, self.decoder_final_state = tf.nn.dynamic_rnn(self.decoder_cell, self.decoder_inputs_embedded,
                                                                 sequence_length=self.decoder_sequence_len,initial_state=self.encoder_final_state,
                                                                 dtype=tf.float32, time_major=True,scope="decoder", )


    def model(self,softmax_sampling_size,learning_rate):
        self.decoder_softmax_weight = tf.get_variable("d_softmax", shape=[self.target_vocab_size, self.decoder_hidden_units],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.decoder_softmax_bias = tf.Variable(tf.random_normal([self.target_vocab_size], stddev=0.1),name="d_bias")

        # sampling softmax cross entropy loss
        # make batch to flat for easy calculation
        self.sampled_softmax_cross_entropy_loss = tf.nn.sampled_softmax_loss(weights=self.decoder_softmax_weight,
                                                                        biases=self.decoder_softmax_bias,
                                                                        labels=tf.reshape(self.decoder_targets, [-1, 1]),
                                                                        inputs=tf.reshape(self.decoder_outputs,
                                                                                          [-1, self.decoder_hidden_units]),
                                                                        num_sampled = softmax_sampling_size,
                                                                        num_classes = self.target_vocab_size, num_true=1)

        self.loss = tf.reduce_mean(self.sampled_softmax_cross_entropy_loss)
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def log_and_saver(self,log_path,model_path,sess):
        # log
        self.loss_sum = tf.summary.scalar("Loss", self.loss)
        self.summary = tf.summary.merge_all()

        self.writer_tr = tf.summary.FileWriter(log_path + "/train", sess.graph)
        self.writer_test = tf.summary.FileWriter(log_path+"/test", sess.graph)

        # saver
        self.dir = os.path.dirname(os.path.realpath(model_path))

    def saver(self):
        self.all_saver = tf.train.Saver()

    def variable_initializer(self,sess):
        sess.run(tf.global_variables_initializer())

    # feed_dict function
    def next_feed(self,source_batch,target_batch):
        self.encoder_inputs_, self.encoder_seq_len_ = hp.batch(source_batch)
        self.decoder_targets_, self.decoder_seq_len_ = hp.batch([(sequence) + [self.target_EOS] for sequence in target_batch])
        self.decoder_inputs_, _ = hp.batch([[self.target_EOS] + (sequence) for sequence in target_batch])

        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.decoder_inputs: self.decoder_inputs_,
            self.decoder_targets: self.decoder_targets_,

            self.source_embeddings: self.source_lookup,
            self.target_embeddings: self.target_lookup,
            self.encoder_sequence_len: self.encoder_seq_len_,
            self.decoder_sequence_len: self.decoder_seq_len_
        }

    def train(self,source_train,source_val,target_train,target_val,batch_size,n_epoch,sess):

        print("Start train !!!!!!!")

        count_t = time.time()

        for i in range(n_epoch):
            for start, end in zip(range(0, len(source_train), batch_size),range(batch_size, len(source_train), batch_size)):

                global_step = i * int(len(source_train) / batch_size) + int(start / batch_size + 1)

                # training
                fd = self.next_feed(source_train[start:end],target_train[start:end])
                s_tr, _, l_tr = sess.run([self.summary, self.train_op, self.loss], feed_dict=fd)
                self.writer_tr.add_summary(s_tr, global_step)

                # validation
                tst_idx = np.arange(len(source_val))
                np.random.shuffle(tst_idx)
                tst_idx = tst_idx[0:batch_size]

                fd_tst = self.next_feed(np.take(source_val,tst_idx,0),np.take(target_val,tst_idx,0))
                s_tst, l_tst = sess.run([self.summary, self.loss], feed_dict=fd_tst)
                self.writer_test.add_summary(s_tst, global_step)

                if start == 0 or int(start / batch_size + 1) % 50 == 0:
                    print("Iter", int(start / batch_size + 1), " Training Loss:", l_tr, "Test loss : ", l_tst)

            if (i + 1) % 100 == 0:
                savename = self.dir + "net-" + str(i + 1) + ".ckpt"
                self.all_saver.save(sess=sess, save_path=savename)

            print("epoch : ", i + 1, "loss : ", l_tr, "Test loss : ", l_tst)

        print("Running Time : ", time.time() - count_t)
        print("Training Finished!!!")

    def load_model(self,model_path,model_name,sess):
        restorename = model_path+"/"+model_name
        self.all_saver.restore(sess,restorename)

    ## Beam Search Part

    def beam_search_options(self,beam_size,max_len):
        self.beam_size = beam_size
        self.max_len = max_len

        # placeholders
        self.beam_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='beam_input')
        self.beam_input_embedded = tf.nn.embedding_lookup(self.target_embeddings, self.beam_input)
        self.state_placeholder = tf.placeholder(tf.float32, [None, self.encoder_hidden_units])

    def BeamSearchDecoder(self):

        # computations for beam
        self.beam_outputs, self.beam_state = tf.nn.dynamic_rnn(self.decoder_cell, self.beam_input_embedded, initial_state=self.state_placeholder,
                                                     dtype=tf.float32, time_major=True, scope="plain_decoder")
        self.beam_outputs = tf.reshape(self.beam_outputs, [-1, self.decoder_hidden_units])
        self.logits = tf.matmul(self.beam_outputs, tf.transpose(self.decoder_softmax_weight)) + self.decoder_softmax_bias
        self.prob_pred, self.word_pred = tf.nn.top_k(tf.nn.softmax(self.logits), k=self.beam_size, sorted=False)

    def next_feed_beam(self,batch, word_input, state_input):
        self.encoder_inputs_, self.seq_len_ = hp.batch(batch)
        self.beam_input_, _ = hp.batch(word_input)
        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.source_embeddings: self.source_lookup,
            self.target_embeddings: self.target_lookup,

            self.encoder_sequence_len: self.seq_len_,
            self.beam_input: self.beam_input_,
            self.state_placeholder: state_input
        }

    def chose_highscores(self,score_mat):
        flat_mat = np.ndarray.flatten(score_mat)
        ix = flat_mat.argsort()[-self.beam_size:][::-1]

        for i in range(len(flat_mat)):
            if i in ix:
                flat_mat[i] = 1
            else:
                flat_mat[i] = 0

        return flat_mat

    def Start_beamsearch(self,input_sentence,sess):

        ## model start

        dead_sample = []
        dead_score = []
        dead_k = len(dead_sample)

        l = 0

        while dead_k == 0 and l < self.max_len:

            if l == 0:
                # initial words : EOS token
                init_words = np.array([[self.target_EOS]] * self.beam_size)

                # initial word for calculating semantics of beam sized
                inputword_stacked = np.array([input_sentence[0] for l in range(self.beam_size)])

                # initial cell and hidden state of decoder
                initial_state = sess.run(self.encoder_final_state, feed_dict=self.next_feed(inputword_stacked,init_words))

                # decoding
                fd = self.next_feed_beam(input_sentence, word_input=init_words, state_input=initial_state)

                w, s, state = sess.run([self.word_pred, self.prob_pred, self.beam_state], feed_dict=fd)

                # update live sample, score
                live_sample = np.array([[w[0][i]] for i in range(self.beam_size)])
                live_scores = np.array([[np.log(s[0][i])] for i in range(self.beam_size)])

            else:

                # Search
                # beam size = batch size
                iter_words = np.array([[w] for w in live_sample[:, -1]])

                # decoding
                fd = self.next_feed_beam(input_sentence, word_input=iter_words, state_input=state)

                w, s, state = sess.run([self.word_pred, self.prob_pred, self.beam_state], feed_dict=fd)

                # calculate candidate score
                cand_scores = live_scores + np.log(s)
                cand_scores_flat = np.ndarray.flatten(cand_scores)

                # find candidate word
                cand_sample = np.array([[live_sample[i]] * self.beam_size for i in range(self.beam_size)])
                cand_sample = cand_sample.reshape([self.beam_size * self.beam_size, -1])

                cand_words = np.array([[w[i, j]] for i in range(self.beam_size) for j in range(self.beam_size)])
                cand_words_list = np.concatenate((cand_sample, cand_words), axis=1)

                # find top beam_size word get mask matrix
                selected_idx = self.chose_highscores(cand_scores).astype(int)
                mask = selected_idx > 0

                # select live score and live sample
                live_scores = np.array([s for s, m in zip(cand_scores_flat, mask) if m]).reshape([self.beam_size, 1])
                live_sample = np.array([s for s, m in zip(cand_words_list, mask) if m]).reshape([self.beam_size, -1])

                # find zombies (needs to die)
                zombie = [s[-1] == self.target_EOS or len(s) >= self.max_len for s in live_sample]

                # add zombies to the dead
                dead_sample += [s for s, z in zip(live_sample, zombie) if z]
                dead_score += [s for s, z in zip(live_scores, zombie) if z]
                dead_k = len(dead_score)

            l += 1

        idx = np.argmax(dead_score)
        answer = dead_sample[idx]

        # remove last EOS token
        if answer[-1] == self.target_EOS:
            answer = np.delete(answer, -1)

        return answer





