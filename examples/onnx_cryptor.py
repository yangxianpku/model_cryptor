import sys
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
import numpy as np
import onnxruntime as ort


onnx_model_file           = '../models/resnet18.onnx'
onnx_model_encrypted_file = "../encrypt_models/resnet18.onnx.crt"
# crypt_key                  = "../keys/license.pem"
crypt_key                  = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  # 指定文件或者直接指定crypt_key的内容


cryptor = CryptorFactory.instance(ModelType.MODEL_ONNX)
cryptor.encrypt(model=onnx_model_file, crypt_key=crypt_key, crypted_model=onnx_model_encrypted_file)
model = cryptor.decrypt(onnx_model_encrypted_file, crypt_key=crypt_key)

dummpy_input  = np.ones(shape=(1, 3, 224, 224)).astype(np.float32)

# 1. 使用解密后的模型推理
sess       = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
out_name   = sess.get_outputs()[0].name
pred       = sess.run([out_name], {input_name: dummpy_input})     # 执行推断
print(pred[0].flatten()[:10])

# 2. 使用原始模型推理
sess      = ort.InferenceSession(onnx_model_file, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
out_name   = sess.get_outputs()[0].name
pred       = sess.run([out_name], {input_name: dummpy_input})     # 执行推断
print(pred[0].flatten()[:10])
# [-0.03913289  0.11446398 -1.7967601  -1.2342987  -0.8190062   0.3239604
#  -2.186605   -1.2876639  -1.90192    -0.73147976]clear
