wdir="/content/drive/MyDrive/ColabNotebooks/superpoint/experiment/saved_models/sp_v6"
# pip install tf2onnx
# python3 -m tf2onnx.convert --saved-model /content/drive/MyDrive/ColabNotebooks/superpoint/experiment/saved_models/sp_v6 --opset 13 --output model.onnx
# import onnx

# from onnx_tf.backend import prepare

# onnx_model = onnx.load("/content/drive/MyDrive/ColabNotebooks/superpoint/SuperPoint/model.onnx")  # load onnx model
# tf_rep = prepare(onnx_model)  # prepare tf representation
# tf_rep.export_graph("/content/drive/MyDrive/ColabNotebooks/superpoint/SuperPoint/")  # export the model
import tensorflow as tf
model = tf.saved_model.load(wdir)