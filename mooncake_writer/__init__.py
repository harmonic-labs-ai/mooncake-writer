"""Mooncake Writer - A library for converting text to hash blocks and vice versa.

This library wraps aiperf's mooncake implementation to provide a simple interface
for converting between text and hash block representations.
"""

from mooncake_writer.rolling_hasher import RollingHasher
from mooncake_writer.writer import MooncakeWriter

__version__ = "0.1.0"
__all__ = ["MooncakeWriter", "RollingHasher"]
