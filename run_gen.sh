python main.py --exp_id gen_exp_id --gpu_no -1 --seed 666 --is_bid --is_attn --rnn_type GRU --model_struct Seq2Seq --rein_epoch 10 --rein_lr 0.001 --train_mode rl_pg --max_diff 5 --pert_mode all --reward dynamic --reward_form cost_red_ratio --victim extend --sel_param parameters --exp_file ./data_resource/heuristic_conf/extend_config.json --db_file ./data_resource/database_conf/db_info.conf --data_load ./data_resource/sample_data/sample_data.pt --model_load ./data_resource/Pre-train_100.pt --colinfo_file ./data_resource/database_conf/colinfo.json --wordinfo_file ./data_resource/vocab/wordinfo.json --schema_file ./data_resource/database_conf/schema.json --batch_size 32 --dropout 0.5 --enc_hidden_size 128 --dec_hidden_size 128 --src_vbs 3040 --tgt_vbs 3040