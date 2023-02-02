import sys
import numpy as np
import tensorrt as trt
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
from   utils.common  import *



trt_model_file           = '../models/resnet18.plan'
trt_model_encrypted_file = "../encrypt_models/resnet18.plan.crt"
# crypt_key                  = "../keys/license.pem"
crypt_key                  = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  # 指定文件或者直接指定crypt_key的内容


cryptor = CryptorFactory.instance(ModelType.MODEL_TENSORRT)
cryptor.encrypt(model=trt_model_file, crypt_key=crypt_key, crypted_model=trt_model_encrypted_file)
model = cryptor.decrypt(trt_model_encrypted_file, crypt_key=crypt_key)

dummpy_input = np.ones(shape=(1,3,224,224)).astype(np.float32).flatten()

# 1. 使用解密后的模型推理
runtime  = trt.Runtime(trt.Logger(trt.Logger.ERROR)) 
engine   = runtime.deserialize_cuda_engine(model)   
context  = engine.create_execution_context() 
inputs, outputs, bindings, stream = allocate_buffers(engine)
np.copyto(inputs[0].host, dummpy_input)
pred = do_inference(context, bindings, inputs, outputs, stream)[0]
print(pred[:10])

# 2. 使用原始模型推理
model    = open(trt_model_file, "rb").read()
runtime  = trt.Runtime(trt.Logger(trt.Logger.ERROR)) 
engine   = runtime.deserialize_cuda_engine(model)   
context  = engine.create_execution_context() 
inputs, outputs, bindings, stream = allocate_buffers(engine)
np.copyto(inputs[0].host, dummpy_input)
pred = do_inference(context, bindings, inputs, outputs, stream)[0]
print(pred[:10])
# [-0.03913065  0.11446576 -1.7967604  -1.234298   -0.81900525  0.32396507
#  -2.1866052  -1.2876656  -1.9019196  -0.7314806 ]