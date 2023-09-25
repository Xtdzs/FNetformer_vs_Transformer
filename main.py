import Transformer
import FNetTransformer
import tensorflow as tf
import pandas as pd
import os
import time

# please check your GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_FNet = False

df = pd.read_csv("datasets/eng_-french.csv")
df = df.iloc[:5000]

en = df['English words/sentences']
fr = df['French words/sentences']

preprocessed_en_seqs, preprocessed_fr_seqs, tokenizer_en, tokenizer_fr = Transformer.preprocess(en, fr, 14, 14)

en_seq_len = preprocessed_en_seqs.shape[1]
fr_seq_len = preprocessed_fr_seqs.shape[1]
en_wordvec_size = len(tokenizer_en.word_index)
fr_wordvec_size = len(tokenizer_fr.word_index)

if use_FNet is False:
    model = Transformer.Transformer(max_seq_len_1=14,
                                    max_seq_len_2=13,
                                    embedding_size=300,
                                    vocab_size_1=fr_wordvec_size + 1,
                                    vocab_size_2=en_wordvec_size + 1,
                                    num_heads=5,
                                    dense_unit_num=512,
                                    num_layers=2)

    model((preprocessed_fr_seqs[:1], preprocessed_en_seqs[:1, :-1]))

    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=["accuracy"])

    start_train = time.time()
    model.fit((preprocessed_fr_seqs, preprocessed_en_seqs[:, :-1]),
              preprocessed_en_seqs[:, 1:, tf.newaxis],
              epochs=40,  # 30
              batch_size=64)
    end_train = time.time()

    print("Training time: ", end_train - start_train)
    print("Training time per epoch: ", (end_train - start_train) / 40)
else:
    model = FNetTransformer.Transformer(max_seq_len_1=14,
                                        max_seq_len_2=13,
                                        embedding_size=300,
                                        vocab_size_1=fr_wordvec_size + 1,
                                        vocab_size_2=en_wordvec_size + 1,
                                        num_heads=5,
                                        dense_unit_num=512,
                                        num_layers=2)

    model((preprocessed_fr_seqs[:1], preprocessed_en_seqs[:1, :-1]))

    model.summary()

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=["accuracy"])

    start_train = time.time()
    model.fit((preprocessed_fr_seqs, preprocessed_en_seqs[:, :-1]),
              preprocessed_en_seqs[:, 1:, tf.newaxis],
              epochs=40,  # 30
              batch_size=64)
    end_train = time.time()

    print("Training time: ", end_train - start_train)
    print("Training time per epoch: ", (end_train - start_train) / 40)
