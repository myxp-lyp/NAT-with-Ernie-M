'''
file1_path = '/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori copy/train.de'
file2_path = '/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori copy/train.en'
output_path = '/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori copy/train'

with open(file1_path, 'r', encoding='utf-8') as file1, open(file2_path, 'r', encoding='utf-8') as file2:
    content = file1.read() + '\n' + file2.read()

with open(output_path, 'w', encoding='utf-8') as output_file:
    output_file.write(content)

'''
import subword_nmt

# Step 1: Train BPE model
code_size = 60000  # Replace with your desired code size
input_text_path = '/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori copy/train_new'
output_model_path = '/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori copy/output_code_model'
subword_nmt.BPE.train(open(input_text_path), open(output_model_path, 'w'), num_symbols=code_size)

# Save the BPE codes to a file
with open('bpe_codes.txt', 'w', encoding='utf-8') as bpe_codes_file:
    bpe_codes_file.write(open(output_model_path).read())
