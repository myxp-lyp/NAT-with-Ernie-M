from random import randint
import subprocess
import sys
import argparse
from tqdm import tqdm
import statistics
import shutil

#--max-target-positions: 400

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--number_runs', help='Number of experiments to run for group', default=5, type=int)
    parser.add_argument('-n', '--name', help='Description for bar argument', required=False)
    parser.add_argument('-d', '--dataset', help='Description for bar argument', required=False)
    args = vars(parser.parse_args())

    scores = []
    score_string = []
    for i in tqdm(range(args["number_runs"])):

        checkpoint_dir = f"{args['name']}-{str(i)}"

        seed = randint(0, 2**32 - 1)
        
        run_string = '''python3 train.py <DATASET> 
--task translation_lev --criterion nat_loss --arch vanilla_nat --label-smoothing 0.1 
--optimizer adam --adam-betas (0.9,0.999) 
--adam-eps 1e-6 --lr-scheduler inverse_sqrt --lr 3e-4 --warmup-updates 4000 --max-tokens 4096 
--dropout 0.1 --max-source-positions 400 --max-target-positions 400 --max-update 300000 --seed <SEED>
--clip-norm 5 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 
--fp16 --apply-bert-init --activation-fn gelu --user-dir dad_plugins --src-embedding-copy 
--pred-length-offset --log-interval 1000 --noise full_mask --share-all-embeddings --choose-data deenmul 
--input-transform --eval-bleu-print-samples --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --weight-decay 0.01 
--eval-bleu --eval-bleu-args {"iter_decode_max_iter":0,"iter_decode_winth_beam":1} 
--eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
--eval-bleu-print-samples 
--curriculum-type at-forward --no-epoch-checkpoints
--decoder-learned-pos --encoder-learned-pos 
--save-dir /data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/<NAME>

--reset-optimizer --patience 4'''.replace("<DATASET>", args["dataset"]).replace("<SEED>", str(seed)).replace("<NAME>", checkpoint_dir)
        cmd = run_string.split()
       
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        for line in iter(p.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))

        run_string = '''
python3 train.py <DATASET> 
		--task translation_lev --criterion nat_loss --arch vanilla_nat --label-smoothing 0.1 
		--optimizer adam --adam-betas (0.9,0.999) 
		--adam-eps 1e-6 --lr-scheduler inverse_sqrt --lr 3e-4 --warmup-updates 4000 --max-tokens 4096 
		--dropout 0.1 --max-source-positions 400 --max-target-positions 400 --max-update 300000 --seed <SEED>
		--clip-norm 5 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 
		--fp16 --apply-bert-init --activation-fn gelu --user-dir dad_plugins --src-embedding-copy 
		--pred-length-offset --log-interval 1000 --noise full_mask --share-all-embeddings --choose-data deenmul 
		--input-transform --eval-bleu-print-samples --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --weight-decay 0.01 
		--eval-bleu --eval-bleu-args {"iter_decode_max_iter":0,"iter_decode_winth_beam":1} 
		--eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
		--eval-bleu-print-samples 
		--curriculum-type at-backward --no-epoch-checkpoints
		--decoder-learned-pos --encoder-learned-pos 
		--save-dir /data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/<NAME>
		
		--reset-optimizer --patience 4'''.replace("<DATASET>", args["dataset"]).replace("<SEED>", str(seed)).replace("<NAME>", checkpoint_dir)
        cmd = run_string.split()
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        for line in iter(p.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))


        run_string = '''
python3 train.py <DATASET> 
		--task translation_lev --criterion nat_loss --arch vanilla_nat --label-smoothing 0.1 
		--optimizer adam --adam-betas (0.9,0.999) 
		--adam-eps 1e-6 --lr-scheduler inverse_sqrt --lr 3e-4 --warmup-updates 4000 --max-tokens 4096 
		--dropout 0.1 --max-source-positions 400 --max-target-positions 400 --max-update 300000 --seed <SEED>
		--clip-norm 5 --encoder-layers 6 --encoder-embed-dim 512 --decoder-layers 6 --decoder-embed-dim 512 
		--fp16 --apply-bert-init --activation-fn gelu --user-dir dad_plugins --src-embedding-copy 
		--pred-length-offset --log-interval 1000 --noise full_mask --share-all-embeddings --choose-data deenmul 
		--input-transform --eval-bleu-print-samples --warmup-init-lr 1e-7 --stop-min-lr 1e-9 --weight-decay 0.01 
		--eval-bleu --eval-bleu-args {"iter_decode_max_iter":0,"iter_decode_winth_beam":1} 
		--eval-tokenized-bleu --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 
		--eval-bleu-print-samples 
		--curriculum-type nat --no-epoch-checkpoints
		--decoder-learned-pos --encoder-learned-pos 
		--save-dir /data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/<NAME>
		
		--reset-optimizer --patience 4'''.replace("<DATASET>", args["dataset"]).replace("<SEED>", str(seed)).replace("<NAME>", checkpoint_dir)
        cmd = run_string.split()
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        for line in iter(p.stdout.readline, b""):
            print(line.rstrip().decode("utf-8"))

        run_string = '''
python3 fairseq_cli/generate.py <DATASET> 
	--path /data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/save_new_seed/<CHECKPOINT_DIR>/checkpoint_best.pt --user-dir dad_plugins 
    --task translation_lev --remove-bpe --max-sentences 20 --source-lang de --target-lang en 
    --quiet --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 --iter-decode-with-beam 1 --gen-subset test --results-path 
    /data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/generation_new_save_seed/<CHECKPOINT_DIR>
'''.replace("<DATASET>", args["dataset"]).replace("<CHECKPOINT_DIR>", checkpoint_dir)
        cmd = run_string.split()
        p = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)

        last_line = ""
        for line in iter(p.stdout.readline, b""):
            last_line = line.rstrip().decode("utf-8")
            print(line.rstrip().decode("utf-8"))

        score_string.append(last_line)
        #shutil.rmtree(f"checkpoints/{checkpoint_dir}")

    for score_line in score_string:
        # example score_line = 'Generate test with beam=5: BLEU4 = 39.48, 73.1/49.4/34.0/23.7 (BP=0.957, ratio=0.957, syslen=12421, reflen=12973)'
        first_half = score_line.split(",")[0]
        second_half = first_half.split(":")[1]
        bleu_score = second_half.split()[-1]
        scores.append(float(bleu_score))

    print("Scores:\n", scores)
    print(f"Mean: {statistics.mean(scores)}, Std: {statistics.stdev(scores)}")