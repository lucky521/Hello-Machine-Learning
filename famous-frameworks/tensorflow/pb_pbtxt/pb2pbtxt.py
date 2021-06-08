import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

def graphdef_to_pbtxt(filename): 
  with open(filename,'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  with open('protobuf.txt', 'w') as fp:
    fp.write(str(graph_def))

def graphdef_to_pbtxt_v2(filename): 
  with  tf.compat.v1.gfile.FastGFile(filename, 'rb') as f:
    data = tf.compat.as_bytes(f.read())
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(data)
  with open('protobuf.txt', 'w') as fp:
    fp.write(str(sm))

graphdef_to_pbtxt_v2('xxx_saved_model.pb')
