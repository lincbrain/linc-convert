from dataclasses import dataclass
import os
from typing import Literal, Annotated
import abc
from cyclopts import Parameter
import numpy as np
import zarr

@dataclass
class _ZarrConfig:
    """
    Parameters
    ----------
    out
        Path to the output Zarr directory [\<INP\>.ome.zarr]
    chunk
        Test
    """
    out: str = ""
    chunk: int = 128
    compressor: str = "blosc"
    compressor_opt: str = "{}"
    shard: list[int | str] | None = None
    version: Literal[2, 3] = 3
    driver: Literal["zarr-python","tensorstore", "zarrita"] = "zarr-python"
    nii: bool = False
    
    def __post_init__(self):
        print(self)

ZarrConfig = Annotated[_ZarrConfig, Parameter(name="*")]

class AbstractZarrIO(abc.ABC):
    
    def __init__(self, config: _ZarrConfig):
        self.config = config

    def __getitem__(self, index):
        pass
    def __setitem__(self, index):
        pass
    def create_dataset(self):
        pass

class ZarrPythonIO(AbstractZarrIO):
    def __init__(self, config: _ZarrConfig, overwrite=True):
        super().__init__(config)
        omz = zarr.storage.DirectoryStore(config.out)
        self.zgroup = zarr.group(store=omz, overwrite=overwrite)
    def create_dataset(self,
                       chunk,
                       dtype,
                       dimension_separator = r"/",
                       order = "F",
                       fill_value = 0,
                       compressor = None):
        
        make_compressor(compressor, **compressor_opt)
        
        pass 
    def __getitem__(self, index):
        return self.zgroup[index]
    
class TensorStoreIO(AbstractZarrIO):
    
    pass



def default_write_config(
    path: os.PathLike | str,
    shape: list[int],
    dtype: np.dtype | str,
    chunk: list[int] = [32],
    shard: list[int] | Literal["auto"] | None = None,
    compressor: str = "blosc",
    compressor_opt: dict | None = None,
    version: int = 3,
) -> dict:
    """
    Generate a default TensorStore configuration.
    Parameters
    ----------
    chunk : list[int]
        Chunk size.
    shard : list[int], optional
        Shard size. No sharding if `None`.
    compressor : str
        Compressor name
    version : int
        Zarr version
    Returns
    -------
    config : dict
        Configuration
    """
    path = UPath(path)
    if not path.protocol:
        path = "file://" / path

    # Format compressor
    if version == 3 and compressor == "zlib":
        compressor = "gzip"
    if version == 2 and compressor == "gzip":
        compressor = "zlib"
    compressor_opt = compressor_opt or {}

    # Prepare chunk size
    if isinstance(chunk, int):
        chunk = [chunk]
    chunk = chunk[:1] * max(0, len(shape) - len(chunk)) + chunk

    # Prepare shard size
    if shard:
        if shard == "auto":
            shard = auto_shard_size(shape, dtype)
        if isinstance(shard, int):
            shard = [shard]
        shard = shard[:1] * max(0, len(shape) - len(shard)) + shard

        # Fix incompatibilities
        shard, chunk = fix_shard_chunk(shard, chunk, shape)

    # ------------------------------------------------------------------
    #   Zarr 3
    # ------------------------------------------------------------------
    if version == 3:
        if compressor and compressor != "raw":
            compressor = [make_compressor_v3(compressor, **compressor_opt)]
        else:
            compressor = []

        codec_little_endian = {"name": "bytes", "configuration": {"endian": "little"}}

        if shard:
            chunk_grid = {
                "name": "regular",
                "configuration": {"chunk_shape": shard},
            }

            sharding_codec = {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": chunk,
                    "codecs": [
                        codec_little_endian,
                        *compressor,
                    ],
                    "index_codecs": [
                        codec_little_endian,
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
            codecs = [sharding_codec]

        else:
            chunk_grid = {"name": "regular", "configuration": {"chunk_shape": chunk}}
            codecs = [
                codec_little_endian,
                *compressor,
            ]

        metadata = {
            "chunk_grid": chunk_grid,
            "codecs": codecs,
            "data_type": np.dtype(dtype).name,
            "fill_value": 0,
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": r"/"},
            },
        }
        config = {
            "driver": "zarr3",
            "metadata": metadata,
        }

    # ------------------------------------------------------------------
    #   Zarr 2
    # ------------------------------------------------------------------
    else:
        if compressor and compressor != "raw":
            compressor = make_compressor_v2(compressor, **compressor_opt)
        else:
            compressor = None

        metadata = {
            "chunks": chunk,
            "order": "F",
            "dtype": np.dtype(dtype).str,
            "fill_value": 0,
            "compressor": compressor,
        }
        config = {
            "driver": "zarr",
            "metadata": metadata,
            "key_encoding": r"/",
        }

    # Prepare store
    config["metadata"]["shape"] = shape
    config["kvstore"] = make_kvstore(path)

    return config


def default_read_config(path: os.PathLike | str) -> dict:
    """
    Generate a TensorStore configuration to read an existing Zarr.
    Parameters
    ----------
    path : PathLike | str
        Path to zarr array.
    """
    path = UPath(path)
    if not path.protocol:
        path = "file://" / path
    if (path / "zarr.json").exists():
        zarr_version = 3
    elif (path / ".zarray").exists():
        zarr_version = 2
    else:
        raise ValueError("Cannot find zarr.json or .zarray file")
    return {
        "kvstore": make_kvstore(path),
        "driver": "zarr3" if zarr_version == 3 else "zarr",
        "open": True,
        "create": False,
        "delete_existing": False,
    }


