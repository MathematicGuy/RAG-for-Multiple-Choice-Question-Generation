# import bitsandbytes as bnb
import torch
# print("BitsAndBytes version:", bnb.__version__)
print(torch.__version__)
# print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# /mnt/d/graduation_project/rag-for-multiple-choice-question-generation/mcq-rag/api
#? bitsandbytes, fastapi doesn't get installed in requirements.txt
#? Pytorch CUDA have to downloadedd manually install "torch==2.6.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu126"
#! CUDA version 11.8 get deprecated
