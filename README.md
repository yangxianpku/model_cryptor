# 深度学习模型加解密工具包

深度学习推理模型通常以文件的形式进行保存，相应的推理引擎通过读取模型文件并反序列化即可进行推理过程. 这样一来，任何获得模型文件的人均可以用于自己的项目中, 这使得模型所有者的权益得不到保障. 本工具包实现了对常见模型文件进行加密存储，获得模型的人需要使用原始加密秘钥对模型进行解密才能进行推理，以使得模型开发者的权益得到保障.

当前已经支持的模型类型有：

```python
@unique
class ModelType(Enum): 
    MODEL_TORCH            = "TorchModelCryptor"            # torch模型
    MODEL_TORCHSCRIPT      = "TorchScriptModelCryptor"      # torch script模型, torch.jit.script或torch.jit.trace保存
    MODEL_ONNX             = "ONNXModelCryptor"             # onnx模型
    MODEL_TENSORRT         = "TensorRTModelCryptor"         # tensorrt模型
    MODEL_TORCH2TRT        = "Torch2TRTModelCryptor"        # torch2trt模型
    MODEL_TENSORFLOW       = "TensorFlowModelCryptor"       # tensorflow模型
    MODEL_TF2TRT           = "TF2TRTModelCryptor"           # tensorflow2tensorrt模型
    MODEL_PADDLE           = "PaddleModelCryptor"           # paddlepaddle模型
    MODEL_PADDLE2TRT       = "Paddle2TRTModelCryptor"       # paddle2tensorrt模型
```

## 1. 安装

### 1.1 依赖安装

参考项目目录下的requirements.txt文件，其中除了torch2trt包外，均可以使用包管理工具进行安装. [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)的安装过程如下:

```bash
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install
```

详细安装过程请参考: https://github.com/NVIDIA-AI-IOT/torch2trt

### 1.2 package安装

该项目暂未配备setup.py安装文件, 使用该工具可直接将model_cryptor目录拷贝至自己的工程后import即可.


## 2. 使用

该工具的使用非常简单，examples目录下给出了每种模型的案例脚本，这里我们以torch和tensorrt模型举例说明:

### 2.1 torch模型加解密

```python
from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
import torch


model_file           = '../models/resnet18.pth'
model_encrypted_file = "../encrypt_models/resnet18.pth.crt"
# crypt_key          = "../keys/license.pem"
# 指定文件或者直接指定crypt_key的内容
crypt_key            = b'xYjPtWOHLXL5tP1vI4PxagJgjNqsnWmLX4a4qd-ROj4='  

# 1. 使用指定的模型类型创建加解密器
cryptor = CryptorFactory.instance(ModelType.MODEL_TORCH) 

# 2. 使用指定的密钥加密模型并保存为加密后的模型文件    
cryptor.encrypt(model=model_file, crypt_key=crypt_key, crypted_model=model_encrypted_file)

# 3. 读取加密过的文件并解密返货模型    
model = cryptor.decrypt(model_encrypted_file, crypt_key=crypt_key, map_location = "cpu")

# 4. 使用解密后的模型推理
dummy_input = torch.ones(size=(1,3,224,224), dtype=torch.float32)
pred        = model(dummy_input)
```



### 2.2 tensorrt模型加解密

```python
import sys
import numpy as np
import tensorrt as trt
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory
from   utils.common  import *

model_file           = '../models/resnet18.plan'
model_encrypted_file = "../encrypt_models/resnet18.plan.crt"
crypt_key            = "../keys/license.pem"


cryptor = CryptorFactory.instance(ModelType.MODEL_TENSORRT)
cryptor.encrypt(model=model_file, crypt_key=crypt_key, crypted_model=model_encrypted_file)
model = cryptor.decrypt(model_encrypted_file, crypt_key=crypt_key)

dummpy_input = np.ones(shape=(1,3,224,224)).astype(np.float32).flatten()

runtime  = trt.Runtime(trt.Logger(trt.Logger.ERROR)) 
engine   = runtime.deserialize_cuda_engine(model)   
context  = engine.create_execution_context() 
inputs, outputs, bindings, stream = allocate_buffers(engine)
np.copyto(inputs[0].host, dummpy_input)
pred = do_inference(context, bindings, inputs, outputs, stream)[0]
print(pred[:10])
```

## 3. 项目版本

|    版本   |  日期  | 更新人  |  更新内容   | 备注  |
|   ----   | ----  |   ----  | ----  |  ----  | 
|  v 0.1  | 2023.02.01  | 杨现  |  新建   | 新增对torch、onnx、torchscript、tensorrt、torch2trt模型的支持  |


## 4. 项目维护

@copyright 杨现 CPIC yangxian-001@cpic.com.cn