import torch
from torch.utils.serialization import load_lua
import numpy as np

if __name__ == '__main__':
    model = load_lua("/media/mathieu/e912e715-2be7-4fa2-8295-5c3ef1369dd0/models/deeptracking/dragon/dragon_finetune6.t7")

    first = np.zeros((1, 4, 150, 150), dtype=np.float32)
    second = np.zeros((1, 4, 150, 150), dtype=np.float32)

    f = torch.from_numpy(first)
    s = torch.from_numpy(second)
    print(model)
    print(model.forward([f, s]))