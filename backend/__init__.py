import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    # Every backend.* submodule needs `import risk_engine` to resolve to
    # src/risk_engine without installing this project as a package. Doing it
    # once here (the package __init__) guarantees it runs before any
    # submodule's own imports, since Python always imports a parent package
    # first.
    sys.path.insert(0, _SRC)
