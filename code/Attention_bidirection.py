####################################################
# RNN-Variational Autoencoder model
#  - Author: Myeongjun Jang
#  - email: xkxpa@korea.ac.kr
#  - git: https://github.com/MJ-Jang
#  - version: Tensorflow ver 1.2.1
####################################################

################################################## 1. Import modules ###################################################
import gensim
import numpy as np
import code.helper as hp
from code.utils import *
import tensorflow as tf
import os
import time
from tensorflow.python.layers import core as layers_core
########################################################################################################################

class Bidirectional_attention_model:

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
        self.attention_hidden_units = source_hidden_dim

    def place_holders(self):
        # input, target shape : [max_time, batch_size]
        # sequence length shape : [batch_size]
        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='encoder_inputs')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_targets')
        self.decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32,name='decoder_inputs')
        self.encoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_sequence_len')
        self.decoder_sequence_len = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_sequence_len')

        # Embedding matrix
        # denote lookup table as placeholder
        self.source_embeddings = tf.placeholder(shape=(self.source_vocab_size, self.input_embedding_size), dtype=tf.float32)
        self.target_embeddings = tf.placeholder(shape=(self.target_vocab_size, self.input_embedding_size), dtype=tf.float32)

        self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.source_embeddings, self.encoder_inputs)
        self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.target_embeddings, self.decoder_inputs)

    # Encoder
    def encoder(self):
        self.encoder_cell_fw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)
        self.encoder_cell_bw = tf.contrib.rnn.GRUCell(self.encoder_hidden_units)

        # only need final state encoder
        # semantic vector is final state of encoder
        bi_outputs, self.encoder_final_state = tf.nn.bidirectional_dynamic_rnn(self.encoder_cell_fw, self.encoder_cell_bw,inputs=self.encoder_inputs_embedded,
                                                           sequence_length=self.encoder_sequence_len, dtype=tf.float32, time_major=True, )

        self.encoder_state_concat = tf.concat(self.encoder_final_state,axis=1)
        encoder_outputs = tf.concat(bi_outputs, -1)
        self.encoder_outputs_trans = tf.transpose(encoder_outputs,[1,0,2])

    # Decoder
    def decoder(self,batch_size):
        self.attention_mech = tf.contrib.seq2seq.BahdanauAttention(num_units=self.encoder_hidden_units,
                                                              memory=self.encoder_outputs_trans,
                                                              memory_sequence_length=self.encoder_sequence_len)
        decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_units)

        self.attent_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=self.attention_mech,
                                                          attention_layer_size=self.attention_hidden_units)

        helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_sequence_len, time_major=True)

        zero_state = self.attent_cell.zero_state(batch_size=batch_size, dtype=tf.float32).clone(cell_state=self.encoder_state_concat)

        self.decoder = tf.contrib.seq2seq.BasicDecoder(self.attent_cell, helper, zero_state)
        self.decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(self.decoder)



    def model(self,softmax_sampling_size,learning_rate):
        self.decoder_softmax_weight = tf.get_variable("d_softmax", shape=[self.target_vocab_size, self.encoder_hidden_units],
                                                 initializer=tf.contrib.layers.xavier_initializer())
        self.decoder_softmax_bias = tf.Variable(tf.random_normal([self.target_vocab_size], stddev=0.1),name="d_bias")

        # sampling softmax cross entropy loss
        # make batch to flat for easy calculation
        self.sampled_softmax_cross_entropy_loss = tf.nn.sampled_softmax_loss(weights=self.decoder_softmax_weight,
                                                                        biases=self.decoder_softmax_bias,
                                                                        labels=tf.reshape(self.decoder_targets, [-1, 1]),
                                                                        inputs=tf.reshape(
                                                                            tf.transpose(self.decoder_outputs[0],[1,0,2]),
                                                                            [-1, self.attention_hidden_units]),
                                                                        num_sampled=2000,
                                                                        num_classes=self.target_vocab_size, num_true=1)

        mask = tf.transpose(tf.sequence_mask(self.decoder_sequence_len, dtype=tf.float32))
        mask = tf.reshape(mask, [-1, 1])
        self.loss = tf.matmul(tf.transpose(tf.reshape(self.sampled_softmax_cross_entropy_loss, [-1, 1])), mask)
        self.total_loss = tf.divide(self.loss[0][0], tf.cast(tf.reduce_sum(self.decoder_sequence_len), dtype=tf.float32))
        self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.total_loss)

    def log_and_saver(self,log_path,model_path,sess):
        # log
        self.loss_sum = tf.summary.scalar("Loss", self.total_loss)
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
        self.decoder_targets_,_  = hp.batch([(sequence) + [self.target_EOS] for sequence in target_batch])
        self.decoder_inputs_, self.decoder_seq_len_ = hp.batch([[self.target_EOS] + (sequence) for sequence in target_batch])

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
                s_tr, _, l_tr = sess.run([self.summary, self.train_op, self.total_loss], feed_dict=fd)
                self.writer_tr.add_summary(s_tr, global_step)

                # validation
                tst_idx = np.arange(len(source_val))
                np.random.shuffle(tst_idx)
                tst_idx = tst_idx[0:batch_size]

                fd_tst = self.next_feed(np.take(source_val,tst_idx,0), np.take(target_val,tst_idx,0))
                s_tst, l_tst = sess.run([self.summary, self.total_loss], feed_dict=fd_tst)
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
    # Beam Search Part

    def Beamsearch_options(self,beam_size,max_len):

        self.beam_size = beam_size
        self.max_len = max_len
        self.beam_input = tf.placeholder(shape=(None, None), dtype=tf.int32, name='beam_input')
        self.beam_input_embedded = tf.nn.embedding_lookup(self.target_embeddings, self.beam_input)

        self.state_placeholder = tf.placeholder(tf.float32, [None, self.decoder_hidden_units])
        self.attention_plc = tf.placeholder(tf.float32, shape=[None, self.attention_hidden_units])
        self.time_plc = tf.placeholder(tf.int32, shape=[])
        self.align_plc= tf.placeholder(tf.float32, shape=[None, None])

        # computations for beam
        zero_state_beam = self.attent_cell.zero_state(batch_size=beam_size, dtype=tf.float32).clone(cell_state=self.encoder_state_concat)
        state = tf.contrib.seq2seq.AttentionWrapperState(cell_state=self.state_placeholder,
                                                         attention=self.attention_plc, time=self.time_plc,
                                                         alignments=self.align_plc,
                                                         alignment_history=())

        helper = tf.contrib.seq2seq.TrainingHelper(self.beam_input_embedded, [1] * beam_size, time_major=True)

        beamdecoder_first = tf.contrib.seq2seq.BasicDecoder(self.attent_cell, helper, zero_state_beam)

        self.beam_outputs_first, self.beam_state_first, _ = tf.contrib.seq2seq.dynamic_decode(beamdecoder_first)
        self.beam_outputs_first = tf.reshape(self.beam_outputs_first[0], [-1,self.attention_hidden_units])
        logits_first = tf.matmul(self.beam_outputs_first, tf.transpose(self.decoder_softmax_weight)) + self.decoder_softmax_bias
        self.prob_pred_first, self.word_pred_first = tf.nn.top_k(tf.nn.softmax(logits_first), k=beam_size, sorted=False)
        
        beamdecoder = tf.contrib.seq2seq.BasicDecoder(self.attent_cell, helper, state)
        self.beam_outputs, self.beam_state, _ = tf.contrib.seq2seq.dynamic_decode(beamdecoder)
        self.beam_outputs = tf.reshape(self.beam_outputs[0], [-1, self.attention_hidden_units])
        logits = tf.matmul(self.beam_outputs,
                           tf.transpose(self.decoder_softmax_weight)) + self.decoder_softmax_bias
        self.prob_pred, self.word_pred = tf.nn.top_k(tf.nn.softmax(logits), k=beam_size, sorted=False)
    

    def next_feed_beam(self,batch, word_input, state_input,attention, time, alignments):
        self.encoder_inputs_, self.seq_len_ = hp.batch(batch)
        self.beam_input_, _ = hp.batch(word_input)
        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.source_embeddings: self.source_lookup,
            self.target_embeddings: self.target_lookup,

            self.encoder_sequence_len: self.seq_len_,
            self.beam_input: self.beam_input_,
            self.state_placeholder: state_input,
            self.attention_plc:attention,
            self.time_plc:time,
            self.align_plc:alignments,
        }

    def next_feed_beam_first(self,batch, word_input):
        self.encoder_inputs_, self.seq_len_ = hp.batch(batch)
        self.beam_input_, _ = hp.batch(word_input)
        return {
            self.encoder_inputs: self.encoder_inputs_,
            self.source_embeddings: self.source_lookup,
            self.target_embeddings: self.target_lookup,

            self.encoder_sequence_len: self.seq_len_,
            self.beam_input: self.beam_input_,
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
                source_sentence = [input_sentence[0] for l in range(self.beam_size)]


                fd = self.next_feed_beam_first(source_sentence, word_input=init_words)
                w, s, state = sess.run([self.word_pred_first, self.prob_pred_first, self.beam_state_first], feed_dict=fd)

                # update live sample, score
                live_sample = np.array([[w[0][i]] for i in range(self.beam_size)])
                live_scores = np.array([[np.log(s[0][i])] for i in range(self.beam_size)])

            else:

                # Search
                # beam size = batch size
                iter_words = np.array([[w] for w in live_sample[:, -1]])

                # decoding
                fd = self.next_feed_beam(source_sentence, word_input=iter_words,state_input=np.array(state[0]),
                                         attention=np.array(state[1]),time=np.array(state[2]),alignments=np.array(state[3]))

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





