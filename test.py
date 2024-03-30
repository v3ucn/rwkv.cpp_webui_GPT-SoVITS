import argparse
import time
import sampling
from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
from tokenizer_util import add_tokenizer_argument, get_tokenizer
from rwkv_cpp import rwkv_world_tokenizer
from typing import List

model_path = "./rwkv-5-h-world-7B-Q5_1.bin"


library = rwkv_cpp_shared_library.load_rwkv_shared_library()
print(f'System info: {library.rwkv_get_system_info_string()}')

print('Loading RWKV model')
model = rwkv_cpp_model.RWKVModel(library, model_path)

tokenizer_decode, tokenizer_encode = get_tokenizer('auto', model.n_vocab)