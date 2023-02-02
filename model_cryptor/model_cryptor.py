import warnings
from   typing  import Tuple, Union
from   enum    import Enum, unique
from   pathlib import Path
from   abc     import ABC, abstractmethod
from   cryptography.fernet import Fernet

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
       

class ModelCryptor(ABC):
    def generate_crypt_key(self, file : Union[Path, str] = Path.cwd()) -> bytes:
        """
            @brief: 加密前, 需要生成相应的加密密钥,
            @param: file 生成的加密密钥保存到文件的路径, 可以为Path和str类型, 如果file为文件夹则会自动
                         创建路径+license.pem的文件, 如果file本身就是文件, 就直接使用该文件名.
        """
        key   = Fernet.generate_key()
        fpath = Path(file) if isinstance(file, str) else file
        fpath = Path(fpath, "license.pem") if fpath.is_dir() else fpath
        with open(str(fpath), "wb") as fw:
            fw.write(key)
        return key

    def read_crypt_key(self, file : Union[Path, str]) -> Tuple[bool, bytes]:
        """
            @brief: 从文件读取密钥内容,用于解密模型, 不能为空 
            @param: file 密钥文件相对或绝对路径, 必须为文件不能为文件夹
            @return bool 密钥文件是否读取成功
                    str  密钥的具体内容, bytes类型
        """
        fpath = Path(file) if isinstance(file, str) else file
        fpath = fpath.resolve()
        if fpath.is_dir() or not fpath.exists():
            return False, b''
        return True, open(str(fpath), 'rb').read()

    def _crypt_key_parser(self, crypt_key: Union[Path, str, bytes]) -> bytes:
        """
            @brief: 从密钥文件解析出密钥内容,如果crypt_key本身就位密钥内容,则直接返回
            @param: crypt_key   密钥文件或bytes类型的密钥内容
            @return 返回密钥内容, bytes类型
        """
        crypt_key_bytes = b''
        if isinstance(crypt_key, bytes):
            crypt_key_bytes = crypt_key
        else:
            try:
                with open(str(crypt_key) if isinstance(crypt_key, Path) else crypt_key, "rb") as fr:
                    crypt_key_bytes =  fr.read()
            except (FileNotFoundError, FileExistsError):
                warnings.warn("Invalid Secret Key Path!", ResourceWarning)
            except Exception:
                warnings.warn("Runtime Error, Exit!", RuntimeWarning)
        return crypt_key_bytes

    def _crypt_key_valid(self, crypt_key: bytes) -> bool:
        """
            @brief: 验证密钥是否为空
        """
        if len(crypt_key) == 0:
            warnings.warn("crypt_key Empty!", ResourceWarning)
            return False
        else:
            return True
                

    @abstractmethod
    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        """
            @brief: 基类抽象方法, 使用指定的crypt_key对model进行加密
            @param: model          原始模型文件路径, 必须为文件
            @param: crypt_key      密钥文件路径(Path, str)或密钥(bytes)内容本身
            @param: crypted_model  加密后的模型保存路径
            @return None         
        """
        pass
        


    @abstractmethod
    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        """
            @brief: 基类抽象方法, 对磁盘上保存的指定加密后的模型进行解密
            @param: model      加密后的磁盘模型文件
            @param: crypt_key  解密密钥文件(Path, str)或者密钥内容(bytes)
            @return object     解密后的内容, 直接保存在内容中, 也可以保存在GPU上(*args, **kwargs参数设置)
        """
        pass
        

class TorchModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        super().__init__()

    def encrypt(self, model: Union[Path, str], crypt_key: Union[Path, str, bytes], 
                crypted_model: Union[Path, str], *args, **kwargs) -> None:
        import torch
        import io
        fmodel = str(model) if isinstance(model, Path) else model

        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return

        fkey            = Fernet(crypt_key_bytes)
        model_src       = torch.load(fmodel, map_location=kwargs.get('map_location', "cpu"))
        byte_obj        = io.BytesIO()              
        torch.save(model_src, byte_obj)
        byte_obj.seek(0)
        model_bytes_src = byte_obj.read()
        model_bytes_dst = fkey.encrypt(model_bytes_src)
        
        try:
            save_fpath = Path(crypted_model) if isinstance(crypted_model, str) else crypted_model
            save_fpath = save_fpath.resolve()
            if save_fpath.is_dir():
                save_fpath = Path(str(save_fpath), Path(fmodel).resolve().parts[-1] + ".crt")
            with open(str(save_fpath), "wb") as fw:
                fw.write(model_bytes_dst)
        except Exception:
            warnings.warn("Encrypted Torch Model Write Error!", ResourceWarning)
            return

    def decrypt(self, model: Union[Path, str], 
                crypt_key: Union[Path, str, bytes], *args, **kwargs) -> object:
        import torch
        import io
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return None
        
        fkey   = Fernet(crypt_key_bytes)
        fmodel = str(model) if isinstance(model, Path) else model
        try:
            with open(fmodel, 'rb') as fr:
                model_buffer = fkey.decrypt(fr.read())
                model_bytes  = io.BytesIO(model_buffer)
                model_bytes.seek(0)
                return torch.load(model_bytes, map_location=kwargs.get('map_location', "cpu"))
        except Exception:
            warnings.warn("Torch Model Decrypt Error, Return None!", ResourceWarning)
            return None

        
class TorchScriptModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        super().__init__()

    def encrypt(self, model: Union[Path, str], crypt_key: Union[Path, str, bytes], 
                crypted_model: Union[Path, str], *args, **kwargs) -> None:
        import torch
        import io
        fmodel = str(model) if isinstance(model, Path) else model
        
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return

        fkey = Fernet(crypt_key_bytes)
        model_src       = torch.jit.load(fmodel, map_location=kwargs.get('map_location', "cpu"))
        byte_obj        = io.BytesIO()              
        torch.jit.save(model_src, byte_obj)
        byte_obj.seek(0)
        model_bytes_src = byte_obj.read()
        model_bytes_dst = fkey.encrypt(model_bytes_src)
        
        try:
            save_fpath = Path(crypted_model) if isinstance(crypted_model, str) else crypted_model
            save_fpath = save_fpath.resolve()
            if save_fpath.is_dir():
                save_fpath = Path(str(save_fpath), Path(fmodel).resolve().parts[-1] + ".crt")
            with open(str(save_fpath), "wb") as fw:
                fw.write(model_bytes_dst)
        except Exception:
            warnings.warn("Encrypted TorchScript Model Write Error, Exit!", ResourceWarning)
            return

    def decrypt(self, model: Union[Path, str], 
                crypt_key: Union[Path, str, bytes], *args, **kwargs) -> object:
        import torch
        import io

        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return None
        
        fkey   = Fernet(crypt_key_bytes)
        fmodel = str(model) if isinstance(model, Path) else model
        try:
            with open(fmodel, 'rb') as fr:
                model_buffer = fkey.decrypt(fr.read())
                model_bytes  = io.BytesIO(model_buffer)
                model_bytes.seek(0)
                return torch.jit.load(model_bytes, map_location=kwargs.get('map_location', "cpu"))
        except Exception:
            warnings.warn("TorchScript Model Decrypt Error, Return None!", ResourceWarning)
            return None


class ONNXModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        super().__init__()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return

        fkey       = Fernet(crypt_key_bytes)
        fmodel_src = Path(model) if isinstance(model, str) else model
        fmodel_dst = Path(crypted_model) if isinstance(crypted_model, str) else crypted_model
        try:
            fmodel_dst = fmodel_dst.resolve()
            if fmodel_dst.is_dir():
                fmodel_dst = Path(str(fmodel_dst),  fmodel_src.resolve().parts[-1] + ".crt")
            with open(str(fmodel_dst), 'wb') as fw:
                crypt_model_bytes = open(str(fmodel_src.resolve()), 'rb').read()
                fw.write(fkey.encrypt(crypt_model_bytes))
        except Exception:
            warnings.warn("Encrypted ONNX Model Write Error, Exit!", ResourceWarning)
            return
        


    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return None

        fkey   = Fernet(crypt_key_bytes)
        fmodel = str(model) if isinstance(model, Path) else model
        try:
            with open(fmodel,'rb') as fr:
                return fkey.decrypt(fr.read())
        except Exception:
            warnings.warn("Decrypt ONNX Model Error, Exit!", RuntimeError)
            return None


class TensorRTModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        super().__init__()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return

        fkey            = Fernet(crypt_key_bytes)
        fmodel          = Path(model) if isinstance(model, str) else model
        fmodel          = fmodel.resolve()
        model_bytes_src = fmodel.read_bytes()
        model_bytes_dst = fkey.encrypt(model_bytes_src)

        try:
            save_fpath = Path(crypted_model) if isinstance(crypted_model, str) else crypted_model
            save_fpath = save_fpath.resolve()
            if save_fpath.is_dir():
                save_fpath = Path(str(save_fpath), Path(fmodel).parts[-1] + ".crt")
            with open(str(save_fpath), "wb") as fw:
                fw.write(model_bytes_dst)
        except Exception:
            warnings.warn("Encrypted TensorRT Model Write Error!", ResourceWarning)
            return
        

    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return None
        
        fkey   = Fernet(crypt_key_bytes)
        fmodel = Path(model) if isinstance(model, str) else model
        fmodel = fmodel.resolve()
        try:
            return fkey.decrypt(fmodel.read_bytes())
        except Exception:
            warnings.warn("TensorRT Model Decrypt Error, Return None!", ResourceWarning)
            return None


class Torch2TRTModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        super().__init__()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        import torch
        import io
        fmodel = str(model) if isinstance(model, Path) else model

        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return

        fkey            = Fernet(crypt_key_bytes)
        model_src       = torch.load(fmodel, map_location=kwargs.get('map_location', "cpu"))
        byte_obj        = io.BytesIO()              
        torch.save(model_src, byte_obj)
        byte_obj.seek(0)
        model_bytes_src = byte_obj.read()
        model_bytes_dst = fkey.encrypt(model_bytes_src)
        
        try:
            save_fpath = Path(crypted_model) if isinstance(crypted_model, str) else crypted_model
            save_fpath = save_fpath.resolve()
            if save_fpath.is_dir():
                save_fpath = Path(str(save_fpath), Path(fmodel).resolve().parts[-1] + ".crt")
            with open(str(save_fpath), "wb") as fw:
                fw.write(model_bytes_dst)
        except Exception:
            warnings.warn("Encrypted Torch2TRT Model Write Error!", ResourceWarning)
            return
        
    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        import torch
        import io
        crypt_key_bytes = self._crypt_key_parser(crypt_key)
        if not self._crypt_key_valid(crypt_key_bytes):
            return None
        
        fkey   = Fernet(crypt_key_bytes)
        fmodel = str(model) if isinstance(model, Path) else model
        try:
            with open(fmodel, 'rb') as fr:
                model_buffer = fkey.decrypt(fr.read())
                model_bytes  = io.BytesIO(model_buffer)
                model_bytes.seek(0)
                from torch2trt import TRTModule
                trt_model = TRTModule()
                trt_model.load_state_dict(torch.load(model_bytes))
                return trt_model
        except Exception:
            warnings.warn("Torch2TRT Model Decrypt Error, Return None!", ResourceWarning)
            return None


class TensorFlowModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        raise NotImplementedError()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        raise NotImplementedError()

    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        raise NotImplementedError()


class TF2TRTModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        raise NotImplementedError()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        raise NotImplementedError()

    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        raise NotImplementedError()


class PaddleModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        raise NotImplementedError()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        raise NotImplementedError()

    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        raise NotImplementedError()


class Paddle2TRTModelCryptor(ModelCryptor):
    def __init__(self) -> None:
        raise NotImplementedError()

    def encrypt(self, model : Union[Path, str], crypt_key : Union[Path, str, bytes],
                crypted_model : Union[Path, str], *args, **kwargs) -> None:
        raise NotImplementedError()

    def decrypt(self, model : Union[Path, str], 
                crypt_key : Union[Path, str, bytes], *args, **kwargs) -> object:
        raise NotImplementedError()



