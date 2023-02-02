import sys
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
import torch


torchscript_model_file           = '../models/recognition-48-cuda.pt'
torchscript_model_encrypted_file = "../encrypt_models/recognition-48-cuda.pt.crt"
# crypt_key                  = "../keys/license.pem"
crypt_key                  = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  # 指定文件或者直接指定crypt_key的内容


cryptor = CryptorFactory.instance(ModelType.MODEL_TORCHSCRIPT)
cryptor.encrypt(model=torchscript_model_file, crypt_key=crypt_key, crypted_model=torchscript_model_encrypted_file)
model = cryptor.decrypt(torchscript_model_encrypted_file, crypt_key=crypt_key, map_location = "cuda:0")

dummy_input = torch.ones(size=(1,3,224,224), dtype=torch.float32).to("cuda:0")

# 1. 使用解密后的模型推理
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])

# 2. 使用原始模型推理
model       = torch.jit.load(torchscript_model_file)
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])
# tensor([18.7077,  1.5036,  3.2877,  0.9628,  0.6669,  1.3666, -0.2074,  1.6376,
#         -0.0676,  1.2917], grad_fn=<SliceBackward0>)