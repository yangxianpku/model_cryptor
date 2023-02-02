import sys
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
import torch
import numpy as np


torch_model_file           = '../models/resnet18.pth'
torch_model_encrypted_file = "../encrypt_models/resnet18.pth.crt"
# crypt_key                  = "../keys/license.pem"
crypt_key                  = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  # 指定文件或者直接指定crypt_key的内容


cryptor = CryptorFactory.instance(ModelType.MODEL_TORCH)
cryptor.encrypt(model=torch_model_file, crypt_key=crypt_key, crypted_model=torch_model_encrypted_file)
model = cryptor.decrypt(torch_model_encrypted_file, crypt_key=crypt_key, map_location = "cpu")

dummy_input = torch.ones(size=(1,3,224,224), dtype=torch.float32)

# 1. 使用解密后的模型推理
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])

# 2. 使用原始模型推理
model       = torch.load(torch_model_file)
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])
# tensor([-0.6471, -0.5629, -0.4566, -1.4785, -0.7065, -0.1057, -0.5283,  0.4873,
#          0.2514, -0.8640], grad_fn=<SliceBackward0>)