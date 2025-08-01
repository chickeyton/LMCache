# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum
from typing import List, TYPE_CHECKING
import abc

from lmcache.utils import CacheEngineKey

if TYPE_CHECKING:
    from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface


class StorageBackendListener(metaclass=abc.ABCMeta):
    """Listener for events happen inside storage backend."""
    def on_evict(self, backend: "StorageBackendInterface", keys: List[CacheEngineKey]):
        pass
