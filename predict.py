import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TextCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab
from tensorflow import saved_model as sm

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = 'model/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

pd_dir = 'model/saved_model'
pd_path = os.path.join(pd_dir, '')

signature_key = sm.signature_constants.CLASSIFY_INPUTS


class CnnModel:
    def __init__(self):
        self.config = TextCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # ckpt 预测方法
        self.model = TextCNN(self.config)
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

        # saved model 预测方法
        # self.meta_graph_def = sm.loader.load(self.session, tags=[sm.tag_constants.SERVING], export_dir=pd_path)
        # signature = self.meta_graph_def.signature_def
        #
        # x_tensor_name = signature[signature_key].inputs['input_x'].name
        # kp_tensor_name = signature[signature_key].inputs['keep_prob'].name
        # y_tensor_name = signature[signature_key].outputs['output'].name
        #
        # self.x = self.session.graph.get_tensor_by_name(x_tensor_name)
        # self.kp = self.session.graph.get_tensor_by_name(kp_tensor_name)
        # self.y = self.session.graph.get_tensor_by_name(y_tensor_name)

    def predict(self, message):
        content = str(message)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)

        # feed_dict = {
        #     self.x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
        #     self.kp: 1.0
        # }
        # y_pred_cls = self.session.run(self.y, feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    test_strs = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
                 '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00',
                 '芒果卫视推出的一款新型综艺节目',
                 '天津市网信办召开2018“瑞犬迎新幸福中华”京津冀网络大过年主题活动推动会',
                 '国足出线了!',
                 '10月27日基金盘前交易提示10月27日基金盘前交易提示：【基金发行开始日】财通价值【基金上市日】广发聚利【基金发行日】博时回报大成消费大成可转债丰利B泰达500中银中小盘天弘丰利鹏华房地产大摩深证300天治稳债中海消费海富通国策建信恒稳汇丰货币A汇丰货币B中邮380民生景气财通价值']
    for i in test_strs:
        print(cnn_model.predict(i))
