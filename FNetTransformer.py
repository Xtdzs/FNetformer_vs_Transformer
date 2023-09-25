import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time


def preprocess(seqs_1, seqs_2, pad_length_1=None, pad_length_2=None):
    tokenizer_1 = Tokenizer()
    tokenizer_2 = Tokenizer()
    tokenizer_1.fit_on_texts(seqs_1)
    tokenizer_2.fit_on_texts(seqs_2)
    preprocessed_1 = tokenizer_1.texts_to_sequences(seqs_1)
    preprocessed_2 = tokenizer_2.texts_to_sequences(seqs_2)
    if pad_length_1 is None:
        pad_length_1 = max([len(sentence) for sentence in preprocessed_1])
    if pad_length_2 is None:
        pad_length_2 = max([len(sentence) for sentence in preprocessed_2])
    preprocessed_1 = pad_sequences(preprocessed_1, maxlen=pad_length_1, padding='post')
    preprocessed_2 = pad_sequences(preprocessed_2, maxlen=pad_length_2, padding='post')

    return preprocessed_1, preprocessed_2, tokenizer_1, tokenizer_2


def set_positional_encoding(max_seq_len, wordvec_size):
    pos = np.arange(max_seq_len).reshape(1, -1).T
    i = np.arange(wordvec_size / 2).reshape(1, -1)
    pos_emb = np.empty((1, max_seq_len, wordvec_size))
    pos_emb[:, :, 0::2] = np.sin(pos / np.power(10000, (2 * i / wordvec_size)))
    pos_emb[:, :, 1::2] = np.cos(pos / np.power(10000, (2 * i / wordvec_size)))

    return tf.cast(pos_emb, dtype=tf.float32)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.positional_code = set_positional_encoding(max_seq_len, embedding_size)

    def call(self, inputs):
        return inputs + self.positional_code


class PaddingMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        padding_mask = 1 - tf.cast(tf.math.equal(inputs, 0), tf.float32)

        return padding_mask[:, tf.newaxis, :]


class LookAheadMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, sequence_length):
        look_ahead_mask = tf.linalg.band_part(tf.ones((1, sequence_length, sequence_length)), -1, 0)

        return look_ahead_mask


class FourierSublayer(layers.Layer):
    def __init__(self, embedding_size, **kwargs):
        super(FourierSublayer, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def call(self, x):
        # According to the paper, we just extract the real part of the fourier transform.
        F_seq = tf.signal.fft(tf.cast(x, tf.complex64))
        F_seq_real = tf.cast(tf.math.real(F_seq), tf.float32)

        x_transposed = tf.transpose(x, perm=[0, 2, 1])

        F_h = tf.signal.fft(tf.cast(x_transposed, tf.complex64))
        F_h = tf.transpose(F_h, perm=[0, 2, 1])

        F_h_real = tf.cast(tf.math.real(F_h), tf.float32)

        output_real = F_h_real + F_seq_real

        return output_real


class InitEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_size, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.padding_mask = PaddingMask()
        self.word_embed = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_seq_len,
                                                    input_shape=(max_seq_len,))
        self.positional_encoder = PositionalEncoding(max_seq_len, embedding_size)

    def call(self, inputs):
        padding_mask = self.padding_mask(inputs)
        embedded_seqs = self.word_embed(inputs)

        return self.positional_encoder(embedded_seqs), padding_mask


class InitDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_size, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.padding_mask = PaddingMask()
        self.look_ahead_mask = LookAheadMask()
        self.word_embed = tf.keras.layers.Embedding(vocab_size, embedding_size, input_length=max_seq_len,
                                                    input_shape=(max_seq_len,))
        self.positional_encoder = PositionalEncoding(max_seq_len, embedding_size)
        self.max_seq_len = max_seq_len

    def call(self, inputs):
        padding_mask = self.padding_mask(inputs)
        embedded_seqs = self.word_embed(inputs)
        look_ahead_mask = self.look_ahead_mask(self.max_seq_len)
        look_ahead_mask = tf.bitwise.bitwise_and(tf.cast(look_ahead_mask, dtype=np.int8),
                                                 tf.cast(padding_mask, dtype=np.int8))

        return self.positional_encoder(embedded_seqs), look_ahead_mask


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, dense_unit_num, dropout_rate=0.0, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.fourier_sublayer = FourierSublayer(embedding_size)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(embedding_size, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.Dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.norm_2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, training):
        fourier_output = self.fourier_sublayer(inputs)
        fourier_output = self.Dropout(fourier_output, training=training)
        norm = self.norm_1(inputs + fourier_output)
        ff_output = self.ff(norm)
        ff_output = self.Dropout(ff_output, training=training)
        output = self.norm_2(norm + ff_output)

        return output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_size, vocab_size, num_heads, dense_unit_num, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.add = tf.keras.layers.Add()
        self.init_layer = InitEncoderLayer(max_seq_len, embedding_size, vocab_size)
        self.encoder_layers = [EncoderLayer(embedding_size, num_heads, dense_unit_num) for _ in range(num_layers)]
        self.num_layers = num_layers

    def call(self, inputs, training):
        final_inputs, mask = self.init_layer(inputs)
        residual_inputs = final_inputs
        for layer in self.encoder_layers:
            final_inputs = layer(final_inputs, training)
            final_inputs = self.add([residual_inputs, final_inputs])
            residual_inputs = final_inputs

        return final_inputs, mask


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, num_heads, dense_unit_num, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.masked_mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_size,
            dropout=dropout_rate,
        )
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embedding_size,
            dropout=dropout_rate,
        )
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(dense_unit_num, activation="relu"),
            tf.keras.layers.Dense(embedding_size, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.Dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.norm_3 = tf.keras.layers.LayerNormalization()

    def call(self, inputs, encoder_output, enc_mask, look_head_mask, training):
        mha_out_1, attention_score_1 = self.masked_mha(inputs, inputs, inputs, look_head_mask, return_attention_scores=True)
        Z_1 = self.norm_1(self.add([inputs, mha_out_1]))
        mha_out_2, attention_score_2 = self.mha(Z_1, encoder_output, encoder_output, enc_mask, return_attention_scores=True)
        Z_2 = self.norm_2(self.add([Z_1, mha_out_2]))
        ff_output = self.ff(Z_2)
        dropped_out = self.Dropout(ff_output, training=training)
        output = self.norm_3(self.add([dropped_out, Z_2]))

        return output


class Decoder(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, embedding_size, vocab_size, num_heads, dense_unit_num, num_layers, **kwargs):
        super().__init__(**kwargs)
        self.add = tf.keras.layers.Add()
        self.init_layer = InitDecoderLayer(max_seq_len, embedding_size, vocab_size)
        self.decoder_layers = [DecoderLayer(embedding_size, num_heads, dense_unit_num) for i in range(num_layers)]
        self.num_layers = num_layers

    def call(self, inputs, encoder_output, enc_mask, training):
        final_inputs, look_head_mask = self.init_layer(inputs)
        residual_inputs = final_inputs
        for layer in self.decoder_layers:
            final_inputs = layer(final_inputs, encoder_output, enc_mask, look_head_mask, training)
            final_inputs = self.add([residual_inputs, final_inputs])
            residual_inputs = final_inputs

        return final_inputs


class Transformer(tf.keras.Model):
    def __init__(self,
                 max_seq_len_1=None,
                 max_seq_len_2=None,
                 embedding_size=None,
                 vocab_size_1=None,
                 vocab_size_2=None,
                 num_heads=None,
                 dense_unit_num=None,
                 num_layers=None):
        super(Transformer, self).__init__()

        self.Encoder = Encoder(max_seq_len_1,
                               embedding_size,
                               vocab_size_1,
                               num_heads,
                               dense_unit_num,
                               num_layers)

        self.Decoder = Decoder(max_seq_len_2,
                               embedding_size,
                               vocab_size_2,
                               num_heads,
                               dense_unit_num,
                               num_layers, )

        self.Final_layer = tf.keras.layers.Dense(vocab_size_2, activation='relu')

        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        input_seqs, output_seqs = inputs
        enc_output, enc_mask = self.Encoder(input_seqs)
        dec_output = self.Decoder(output_seqs, enc_output, enc_mask)
        final_out = self.Final_layer(dec_output)
        softmax_out = self.softmax(final_out)

        return softmax_out



