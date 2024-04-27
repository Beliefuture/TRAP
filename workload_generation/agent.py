import os
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from workload_generation.environ import DBEnviron
from workload_generation.generation_utils import gen_com
from workload_generation.model import Seq2Seq, Actor, \
    CrossEntropy, SelfCriticCriterion, SingleGRUModel


class WorkloadGeneration:
    def __init__(self, args):
        self.args = args
        self.env = DBEnviron(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pre_train(self, train_loader, valid_loader,
                  word2idx, idx2word, col_info, word_info):
        model = Seq2Seq(self.args)
        criterion = CrossEntropy()  # nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), self.args.pre_lr)
        scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.5,
                                      patience=20, min_lr=1e-5, verbose=True)

        model = model.to(self.device)
        criterion = criterion.to(self.device)

        for epoch in tqdm(range(1, self.args.pre_epoch + 1)):
            logging.info(f"The `lr` of EP{epoch} is `{optimizer.param_groups[0]['lr']}`.")

            model.train()
            total_loss = 0
            pro_bar = tqdm(enumerate(train_loader))
            for bi, batch in pro_bar:
                pro_bar.set_description(f"Epoch [{epoch}/{self.args.pre_epoch}]")
                optimizer.zero_grad()

                tensor_src, tensor_tgt, sql_tokens = batch
                tensor_src, tensor_tgt = tensor_src.to(self.device), tensor_tgt.to(self.device)

                props, _ = model(tensor_src, tensor_tgt, self.device,
                                 word2idx, idx2word, word_info, col_info,
                                 sql_tokens=None, teacher_forcing_ratio=self.args.force_ratio,
                                 max_diff=self.args.max_diff)
                loss = criterion(props, tensor_tgt, self.device)
                
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pro_bar.set_postfix(train_loss=total_loss / (bi + 1))

                gen_com.add_summary_value("pre-train loss", loss.item())
                gen_com.tf_step += 1
                if gen_com.tf_step % 100 == 0:
                    gen_com.summary_writer.flush()
            logging.info(f"The final train loss of EP{epoch} is: {total_loss / (bi + 1)}.")

            model.eval()
            total_loss = 0
            pro_bar = tqdm(enumerate(valid_loader))
            for bi, batch in pro_bar:
                pro_bar.set_description(f"Epoch [{epoch}/{self.args.pre_epoch}]")
                tensor_src, tensor_tgt, sql_tokens = batch
                if torch.cuda.is_available():
                    tensor_src, tensor_tgt = tensor_src.cuda(), tensor_tgt.cuda()

                props, _ = model(tensor_src, tensor_tgt, self.device,
                                 word2idx, idx2word, word_info, col_info, sql_tokens=None,
                                 teacher_forcing_ratio=self.args.force_ratio, max_diff=self.args.max_diff)
                loss = criterion(props, tensor_tgt, self.device)

                total_loss += loss.item()
                pro_bar.set_postfix(valid_loss=total_loss / (bi + 1))

                gen_com.add_summary_value("pre-train valid loss", loss.item())
                gen_com.tf_step += 1
                if gen_com.tf_step % 100 == 0:
                    gen_com.summary_writer.flush()

            scheduler.step(total_loss / (bi + 1))

            logging.info(f"The final valid loss of EP{epoch} is: {total_loss / (bi + 1)}.")

            model_state_dict = model.state_dict()
            model_source = {
                "settings": self.args,
                "model": model_state_dict,
                "word2idx": word2idx,
                "idx2word": idx2word,
                "col_info": col_info
            }
            if epoch % self.args.model_save_gap == 0:
                torch.save(model_source, self.args.model_save.format(
                    self.args.exp_id, "Pre-train_" + str(epoch)))

    def pg_train(self, train_loader, valid_loader,
                 word2idx, idx2word, col_info, word_info):
        if os.path.exists(self.args.model_load):
            model_source = torch.load(self.args.model_load,
                                      map_location=lambda storage, loc: storage)
            if self.args.model_struct == "Seq2Seq":
                model = Actor(self.args)
            elif self.args.model_struct == "SingleRNN":
                model = SingleGRUModel(self.args)

            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in model_source["model"].items()
                               if k in model_dict and "encoder" in k}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            logging.disable(logging.DEBUG)
            logging.info(f"Load the pretrained model from `{self.args.model_load}`.")
            logging.info(f"The set of the pretrained parameters loaded is {list(pretrained_dict.keys())}.")
            logging.disable(logging.INFO)
        else:
            if self.args.model_struct == "Seq2Seq":
                model = Actor(self.args)
            elif self.args.model_struct == "SingleRNN":
                model = SingleGRUModel(self.args)
        logging.disable(logging.DEBUG)
        logging.info(f"The type of the model structure is `{self.args.model_struct}`.")
        logging.disable(logging.INFO)

        criterion = SelfCriticCriterion()  # nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), self.args.rein_lr)
        scheduler = ReduceLROnPlateau(optimizer, "max", factor=0.5,
                                      patience=20, min_lr=1e-5, verbose=True)

        model = model.to(self.device)
        criterion = criterion.to(self.device)

        for epoch in tqdm(range(1, self.args.rein_epoch + 1)):
            logging.disable(logging.DEBUG)
            logging.info(f"The `lr` of EP{epoch} is `{optimizer.param_groups[0]['lr']}`.")
            logging.disable(logging.INFO)

            model.train()
            total_loss, total_reward, total_base = 0, 0, 0
            pro_bar = tqdm(enumerate(train_loader))
            for bi, batch in pro_bar:
                pro_bar.set_description(f"Epoch [{epoch}/{self.args.rein_epoch}]")
                optimizer.zero_grad()

                tensor_src, _, sql_tokens = batch
                tensor_src = tensor_src.to(self.device)
                batch_size = tensor_src.size(0)

                greedy_words = model(tensor_src, self.device,
                                     word2idx, idx2word, word_info, col_info,
                                     sql_tokens, True, max_diff=self.args.max_diff)
                sample_words, samlog_props = model(tensor_src, self.device,
                                                   word2idx, idx2word, word_info, col_info,
                                                   sql_tokens, False, max_diff=self.args.max_diff)

                rewards, baseline = list(), list()
                for qi in range(batch_size):
                    rewards.append(torch.tensor(
                        self.env.get_index_reward(sql_tokens[qi],
                                                  sample_words[qi].cpu().numpy(),
                                                  idx2word, col_info)))
                    baseline.append(torch.tensor(
                        self.env.get_index_reward(sql_tokens[qi],
                                                  greedy_words[qi].cpu().numpy(),
                                                  idx2word, col_info)))

                rewards = torch.stack(rewards, 0).to(self.device)
                baseline = torch.stack(baseline, 0).to(self.device)
                advantage = rewards - baseline

                loss = criterion(samlog_props, sample_words,
                                 tensor_src, advantage, self.device)

                total_loss += loss.item()
                pro_bar.set_postfix(rein_loss=total_loss / (bi + 1))

                total_reward += rewards.mean().item()
                pro_bar.set_postfix(reward=total_reward / (bi + 1))

                total_base += baseline.mean().item()
                pro_bar.set_postfix(baseline=total_base / (bi + 1))

                loss.backward()
                optimizer.step()

                gen_com.add_summary_value("reinforce loss", loss.item())
                gen_com.add_summary_value("reinforce advantage", advantage.mean().item())
                gen_com.add_summary_value("reinforce baseline", baseline.mean().item())
                gen_com.add_summary_value("reinforce reward", rewards.mean().item())
                gen_com.tf_step += 1

                # Flushes the event file to disk.
                # Call this method to make sure that all pending events
                # have been written to disk.
                if gen_com.tf_step % 100 == 0:
                    gen_com.summary_writer.flush()

            gen_com.add_summary_value("epoch reinforce loss", total_loss / (bi + 1), epoch)
            gen_com.add_summary_value("epoch reinforce reward", total_reward / (bi + 1), epoch)
            gen_com.add_summary_value("epoch reinforce baseline", total_base / (bi + 1), epoch)

            logging.disable(logging.DEBUG)
            logging.info(f"The final loss / reward / baseline of EP{epoch} "
                         f"is: {total_loss / (bi + 1)} / {total_reward / (bi + 1)} / {total_base / (bi + 1)}.")
            logging.disable(logging.INFO)

            model_state_dict = model.state_dict()
            model_source = {
                "settings": self.args,
                "model": model_state_dict,
                "word2idx": word2idx,
                "idx2word": idx2word,
                "col_info": col_info
            }
            if epoch % self.args.model_save_gap == 0:
                torch.save(model_source, self.args.model_save.format(
                    self.args.exp_id, "PG-train_" + str(epoch)))
