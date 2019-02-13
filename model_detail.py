import os

import tensorflow as tf
from tensorflow import saved_model as sm

pd_path = 'model/saved_model'
#pd_path = os.path.join(pd_dir, 'best_validation')

session = tf.Session()
session.run(tf.global_variables_initializer())
meta_graph_def = sm.loader.load(session, export_dir=pd_path)
graph_def = session.graph_def

for node in graph_def.node:
    print(node.name)
