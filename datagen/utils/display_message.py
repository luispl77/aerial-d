import os
import time
import setproctitle
import torch

# Set the custom process name that will be visible in nvidia-smi
setproctitle.setproctitle("using 4 cpus, sorry - u035679")

# Allocate a small tensor on GPU to make the process visible in nvidia-smi
if torch.cuda.is_available():
    # Allocate about 100MB of GPU memory
    tensor = torch.zeros((100, 100, 100), device='cuda')
    print(f"Allocated {tensor.numel() * tensor.element_size() / 1024 / 1024:.2f}MB on GPU")
else:
    print("No GPU available!")

# Keep the process running
while True:
    time.sleep(1) 