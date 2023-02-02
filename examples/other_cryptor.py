import sys
sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory

# raise NotImplementedError()
cryptor = CryptorFactory.instance(ModelType.MODEL_TENSORFLOW)

