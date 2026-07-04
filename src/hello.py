"""Environment smoke test."""
import sys, platform
import numpy as np
import pandas as pd
import matplotlib
import torch

print("=" * 55)
print(f"Host        : {platform.node()}")
print(f"Python      : {sys.version.split()[0]}")
print(f"Executable  : {sys.executable}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"Torch       : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
print(f"CPU threads : {torch.get_num_threads()}")
print("=" * 55)

x = torch.randn(3, 3)
y = torch.matmul(x, x.T)
print(f"\nRandom 3x3 · its transpose:\n{y.numpy()}")
