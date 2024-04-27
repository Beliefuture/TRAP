import os
import json
import logging
import random
import numpy as np

import sys
sys.path.append(".")
sys.path.append("./heuristic_advisor")
sys.path.append("./learning_advisor")

from workload_generation.generation_utils import gen_com

# todo: 1. get the params.
parser = gen_com.get_parser()
args = parser.parse_args()

# todo: 2. set the gpu device.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_no  # default: "0"
logging.info(f"Set the gpu_no = `{args.gpu_no}`.")

import torch
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter

from workload_generation.dataset import SQLDataset, collate_fn4sql
from workload_generation.agent import WorkloadGeneration

# todo: 3. create the directory to store the `exp_res`.
# assert not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))), \
#     f"`{os.path.dirname(args.logdir.format(args.exp_id))}` dir already existed! " \
#     f"And we don't intend to overwrite anything."

if not os.path.exists(os.path.dirname(args.logdir.format(args.exp_id))):
    os.makedirs(os.path.dirname(args.logdir.format(args.exp_id)))
if not os.path.exists(os.path.dirname(args.model_save.format(args.exp_id, 0))):
    os.makedirs(os.path.dirname(args.model_save.format(args.exp_id, 0)))
if not os.path.exists(os.path.dirname(args.data_save.format(args.exp_id, 0))):
    os.makedirs(os.path.dirname(args.data_save.format(args.exp_id, 0)))

gen_com.set_logger(args.runlog.format(args.exp_id))
logging.info("Start Adversarial Workload Generation.")

logging.info(f"Create the directory `{os.path.dirname(args.logdir.format(args.exp_id))}` to save experiment result.")

# specify the path to store the exp_res of `logdir` of the tensorboard..
# visrew_com.tf_summary_writer = tf.summary.create_file_writer(args.logdir.format(args.exp_id))
gen_com.summary_writer = SummaryWriter(args.logdir.format(args.exp_id))
gen_com.summary_writer.add_text(
        "parameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        0
    )
logging.info(f"Set the tensorboard logdir = `{args.logdir.format(args.exp_id)}`.")

# todo: 4. set the torch random_seed.
# Sets the seed for generating random numbers.
# Returns a `torch.Generator` object.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
logging.info(f"Set the random seed = `{args.seed}`.")

# todo: 5. load the training data.
data = torch.load(args.data_load)
data["src_vectors"] = [item["pno_tokens"] for item in data["src_tokens"]]
data["tgt_vectors"] = [item["pno_tokens"] for item in data["tgt_tokens"]]
logging.info(f"Load the data from `{args.data_load}({len(data['src_vectors'])})`.")

# todo: 6. split the data and create the train/valid data loader.
if args.pre_mode == "ae":
    dataset = SQLDataset(data["tgt_vectors"], data["tgt_vectors"], data["src_tokens"])
    logging.info(f"All the training data is in the form of `(src, src)`.")
else:
    dataset = SQLDataset(data["src_vectors"], data["tgt_vectors"], data["src_tokens"])
    logging.info(f"All the training data is in the form of `(src, tgt)`.")

train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn4sql, drop_last=True)
valid_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_fn4sql, drop_last=True)

torch.save(dataset, args.data_save.format(args.exp_id, "all"))
logging.info(f"Save the dataset into `{os.path.dirname(args.data_save.format(args.exp_id, 0))}`.")

# todo: 7. start the training.
agent = WorkloadGeneration(args)
logging.info(f"Load the value of `is_bid`({args.is_bid}), `is_attn`({args.is_attn}), "
             f"`is_ptr`({args.is_ptr}), `rnn_type`({args.rnn_type}).")

with open(args.colinfo_file, "r") as rf:
    col_info = json.load(rf)
with open(args.wordinfo_file, "r") as rf:
    word_info = json.load(rf)

if args.train_mode == "pre_train":
    logging.info("Start the `pre_train` mode training.")
    logging.info(f"The teacher forcing ratio is `{args.force_ratio}`.")
    agent.pre_train(train_loader, valid_loader,
                    data["word2idx"], data["idx2word"], col_info, word_info)
    logging.info("End the `pre_train` mode training.")

elif args.train_mode == "rl_pg":
    logging.info("Start the `rl_pg` mode training.")
    logging.disable(logging.INFO)
    agent.env.setup()

    agent.pg_train(train_loader, valid_loader,
                   data["word2idx"], data["idx2word"], col_info, word_info)
    agent.env.connector.close()
    logging.disable(logging.DEBUG)
    logging.info("End the `rl_pg` mode training.")

gen_com.summary_writer.close()
logging.info("Close the tensorboard summary writer.")
