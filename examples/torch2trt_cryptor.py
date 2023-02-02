import sys
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
from   torch2trt     import TRTModule
import torch


torch_model_file           = '../models/resnet18.ptrt'
torch_model_encrypted_file = "../encrypt_models/resnet18.ptrt.crt"
# crypt_key                  = "../keys/license.pem"
crypt_key                  = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  # 指定文件或者直接指定crypt_key的内容


cryptor = CryptorFactory.instance(ModelType.MODEL_TORCH2TRT)
cryptor.encrypt(model=torch_model_file, crypt_key=crypt_key, crypted_model=torch_model_encrypted_file)
model = cryptor.decrypt(torch_model_encrypted_file, crypt_key=crypt_key)

device      = torch.device("cuda:0")
dummy_input = torch.ones(size=(1,3,224,224), dtype=torch.float32).to(device)

# 1. 使用解密后的模型推理
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])

# 2. 使用原始模型推理
model       = TRTModule()
model.load_state_dict(torch.load(torch_model_file))
pred        = model(dummy_input)
print(pred.cpu().flatten()[:10])
# tensor([-0.8870, -0.3161, -1.4595, -1.2644, -0.6855,  0.1806, -1.1382, -0.9406,
#         -1.7186, -0.7356], device='cuda:0')