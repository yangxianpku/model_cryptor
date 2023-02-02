import sys

sys.path.append("..")

from   model_cryptor import ModelType
from   model_cryptor import CryptorFactory


crypt_key_file = "license.pem"
cryptor   = CryptorFactory.instance(ModelType.MODEL_TORCH)
crypt_key = cryptor.generate_crypt_key(crypt_key_file)

print(crypt_key)


status, crypt_key = cryptor.read_crypt_key(crypt_key_file)
print(status, crypt_key)