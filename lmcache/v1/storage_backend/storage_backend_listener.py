from enum import Enum
from typing import List, TYPE_CHECKING
import abc

from lmcache.utils import CacheEngineKey

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface


class StorageBackendListener(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _setup_backend_listener(self) -> None:
        """
        Set up listener for all backends.
        """
        raise NotImplementedError

    def on_evict(self, backend: "StorageBackendInterface", keys: List[CacheEngineKey]):
        pass