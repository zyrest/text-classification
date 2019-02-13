import os

import numpy as np
import tensorflow as tf
from sklearn import metrics
from tensorflow import saved_model as sm
from cnn_model import TextCNNConfig
from data.cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab


base_dir = 'data/cnews'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

pd_dir = 'model/saved_model'
pd_path = os.path.join(pd_dir, 'best_validation')

signature_key = sm.signature_constants.CLASSIFY_INPUTS


def test():
    print("Loading test data...")
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    meta_graph_def = sm.loader.load(session, tags=[sm.tag_constants.TRAINING], export_dir=pd_path)
    signature = meta_graph_def.signature_def

    x_tensor_name = signature[signature_key].inputs['input_x'].name
    y_tensor_name = signature[signature_key].inputs['input_y'].name
    kp_tensor_name = signature[signature_key].inputs['keep_prob'].name
    out_tensor_name = signature[signature_key].outputs['output'].name
    acc_tensor_name = signature[signature_key].outputs['acc'].name
    loss_tensor_name = signature[signature_key].outputs['loss'].name

    x = session.graph.get_tensor_by_name(x_tensor_name)
    y = session.graph.get_tensor_by_name(y_tensor_name)
    kp = session.graph.get_tensor_by_name(kp_tensor_name)
    out = session.graph.get_tensor_by_name(out_tensor_name)
    acc = session.graph.get_tensor_by_name(acc_tensor_name)
    loss = session.graph.get_tensor_by_name(loss_tensor_name)

    print(out_tensor_name)
    print('Testing...')

    data_len = len(x_test)
    batch_eval = batch_iter(x_test, y_test, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = {x: x_batch, y: y_batch, kp: 1.0}
        loss_, acc_ = session.run([loss, acc], feed_dict=feed_dict)
        total_loss += loss_ * batch_len
        total_acc += acc_ * batch_len
    loss_test,  acc_test = total_loss / data_len, total_acc / data_len
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            x: x_test[start_id:end_id],
            kp: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(out, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)


config = TextCNNConfig()
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
test()
