# TRAP

This is the code respository of **TRAP**, which is a unified framework concerning the robustness assessment over various index advisors. Specifically, the assessment is conducted based on **the perturbation-based adversarial workloads** according to the observations from the open-source benchmarks and real world workloads. 

The repository contains the following contents:

1. **Running Script:** The scripts for the experiment setup and the workload generation.

2. **Constraint-Aware Reference Tree:** The additional information, i.e., the BNF Grammar, the token type and the vocabulary.

3. **Cost estimation:** The cost estimation results, i.e., the learned model's accuaracy over different models with various feature channels.

   

## 1.1 Setup

We introduce the indispensable step, i.e., experiment setup before workload generation in **TRAP**, you should check the following things:

- Create **the TPC-H/TPC-DS database instance** according to the toolkit provided in [TPC-Homepage](https://www.tpc.org/);
- Create **the HypoPG extension** on the TPC-H/TPC-DS database instance for the usage of hypothetical index according to [HypoPG/hypopg: Hypothetical Indexes for PostgreSQL (github.com)](https://github.com/HypoPG/hypopg);
- Create **the python virtual environment** tailored for **TRAP**. Specifically, you can utilize the following script and the corresponding file `requirements.txt` is provided under the main directory. Please check the packages required are properly installed.

```shell
# Create the virtualenv `TRAP`
conda create -n TRAP python=3.7		 	

# Activate the virtualenv `TRAP`
conda activate TRAP				

# Install requirements with pip
while read requirement; do pip install $requirement; done < requirements.txt	
```



## 1.2 Module List

The description of files in each module is shown below.

| Module              | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| data_resource       | The configuration over database (e.g., the user name), heuristic-based and learning-based index advisors (e.g., the tuning constraint), the global SQL token's vocabulary and the sample data leveraged during the generation of workload. |
| heuristic_advisor   | The implementation of heuristic-based index advisors.        |
| learning_advisor    | The implementation of learning-based index advisors.         |
| workload_generation | The implementation of adversarial workload generation module. |



## 1.3 Adversarial Workload Generation

You should make sure **the packages** required in the `requirements.txt` are properly installed and connect to your own **database instance** to specify the configuration in `/data_source/database_conf/db_info.conf`!!!

```shell
python main.py 
--exp_id gen_exp_id --gpu_no -1 --seed 666  
--is_bid --is_attn --rnn_type GRU --model_struct Seq2Seq 

--rein_epoch 10 --rein_lr 0.001 --train_mode rl_pg 
--max_diff 5 --pert_mode all 
--reward dynamic --reward_form cost_red_ratio 

--victim extend --sel_param parameters 
--exp_file ./data_resource/heuristic_conf/extend_config.json  

--db_file ./data_resource/database_conf/db_info.conf 
--data_load ./data_resource/sample_data/sample_data.pt 

--model_load ./data_resource/Pre-train_100.pt  
--colinfo_file ./data_resource/database_conf/colinfo.json 
--wordinfo_file ./data_resource/vocab/wordinfo.json 
--schema_file ./data_resource/database_conf/schema.json  

--batch_size 32 --dropout 0.5 
--enc_hidden_size 128 --dec_hidden_size 128 
--src_vbs 3040 --tgt_vbs 3040
```



## 2. BNF Grammar, Token Type and Vocabulary

The ***BNF Grammar*** for a standard **Select-Project-Aggregate-Join (SPAJ)** query is presented in the following table. Please refer to **the directory, i.e., `/data_source/vocab/*`** for the details of the token type and the vocabulary.

| SQL ::= SELECT FROM [WHERE] [GROUPBY] [HAVING] [ORDERBY] <br>SELECT ::= "select" (term ("," term)? \| SQL) <br/>FROM ::= "from" (table (("," table)? \| ("join" table)?) \| SQL) <br/>WHERE ::= "where" predicate1 (conj predicate1)? <br/>GROUPBY ::= "group by" column ("," column)? <br/>HAVING ::= "having" predicate2 ("," predicate2)? <br/>ORDERBY ::= "order by" (term \| term key) ("," term \| "," term key)? |
| ------------------------------------------------------------ |
| term ::= column \| agg "(" column ")" <br/>predicate1 ::= column op (value \| SQL) <br/>predicate2 ::= term op (value \| SQL) <br/>table ::= "t1" \| "t2" <br/>column ::= "t1.col1" \| "t1.col2" \| "t2.col1" <br/>agg ::= "min" \| "max" \| "count" \| "sum" \| "avg" <br/>op ::= ">" \| "<" \| ">=" \| "<=" \| "!=" <br/>conj ::= "and" \| "or"value ::= "value1" \| "value2" <br/>key ::= "asc" \| "desc" |


