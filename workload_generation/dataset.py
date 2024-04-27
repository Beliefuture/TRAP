import torch
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import torch.nn.functional as F

from workload_generation.generation_utils import constants


def padd_to_longest(raw_data, max_len=55):
    padd_array = np.array([dat + [constants.EOS] +
                           [constants.PAD] * (max_len - len(dat) - 1)
                           for dat in raw_data])

    # numpy array -> tensor
    padd_data_tensor = torch.from_numpy(padd_array)
    return padd_data_tensor


def collate_fn4sql(samples):
    src_sql, tgt_sql, sql_tokens = map(list, zip(*samples))
    padd_tensor_src = padd_to_longest(src_sql, max_len=55)
    padd_tensor_tgt = padd_to_longest(tgt_sql, max_len=55)

    return padd_tensor_src, padd_tensor_tgt, sql_tokens


class SQLDataset(Dataset):
    def __init__(self, src_sql, tgt_sql, sql_tokens):
        self.src_sql = src_sql
        self.tgt_sql = tgt_sql
        self.sql_tokens = sql_tokens

    def __getitem__(self, index):
        return self.src_sql[index], self.tgt_sql[index], self.sql_tokens[index]

    def __len__(self):
        return len(self.src_sql)
