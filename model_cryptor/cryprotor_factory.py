import threading
from   . import model_cryptor
from   model_cryptor.model_cryptor import ModelType
from   model_cryptor.model_cryptor import ModelCryptor

class CryptorFactory():
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # singleton mode, so this method is empty
        pass

    @classmethod
    def instance(cls, type : ModelType,   *args, **kwargs) -> ModelCryptor:
        if (not hasattr(CryptorFactory, "_type")) or (getattr(CryptorFactory, "_type") != type.value):
            with CryptorFactory._instance_lock:
                CryptorFactory._type     = type.value
                CryptorFactory._instance = getattr(model_cryptor, CryptorFactory._type)()
        return CryptorFactory._instance
