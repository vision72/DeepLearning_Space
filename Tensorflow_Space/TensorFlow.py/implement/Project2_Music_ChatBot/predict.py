import tensorflow as tf
import data_utils
import seq2seq_model
import os

def train():
    #prepare dataset
    enc_train, dec_train = data_utils.prepare_custom_data(gConfig['working_directory'])
    train_set = read_data(enc_train, dec_train)

def seq2seq_f(encd_inputs, decd_inputs, decode_thm):
    return tf.nn.seq2seq.embedding_attention_seq2seq(encd_inputs, decd_inputs, cell, num_encd_symb=source_vocab_size, num_decd_symb=target_vocab_size, embedding_size=size, output_projection=output_projection, feed_previous=decode_thm)

with tf.Session() as sess:
    

    while True:
        sess.run(model)

        chck = os.path.join(gConfig['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, chck, global_step=model.global_step)
