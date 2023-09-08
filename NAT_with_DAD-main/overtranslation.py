import os

count=-1
a = []
with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/generation_new_save_seed/wmt_dad_CAMLM_1-0/generate-test.txt', 'r') as f:
    #with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/bert-score/wmt_camlm/wmt_camlm-2_hyps.txt','w') as ff:
    for lines in f:
        if lines[0] == 'D':
            line = lines.split()
            for x in line:
                a.append(x)

word = a[0]
for x in a:
    if x == word:
        count += 1
    else:
        word = x
                
print(count)

count3=-1
a = []
with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/generation_new_save_seed/wmt_dad_LBPE-1/generate-test.txt', 'r') as f:
    #with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/bert-score/wmt_camlm/wmt_camlm-2_hyps.txt','w') as ff:
    for lines in f:
        if lines[0] == 'D':
            line = lines.split()
            for x in line:
                a.append(x)

word = a[0]
for x in a:
    if x == word:
        count3 += 1
    else:
        word = x
                
print(count3)

count1=-1
a = []
with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/generation_new_save_seed/wmt_dad_CAMLMLBPE-0/generate-test.txt', 'r') as f:
    #with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/bert-score/wmt_camlm/wmt_camlm-2_hyps.txt','w') as ff:
    for lines in f:
        if lines[0] == 'D':
            line = lines.split()
            for x in line:
                a.append(x)

word = a[0]
for x in a:
    if x == word:
        count1 += 1
    else:
        word = x
                
print(count1)

count2=-1
a = []
with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/generation_new_save_seed/wmt_dad-0/generate-test.txt', 'r') as f:
    #with open('/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/bert-score/wmt_camlm/wmt_camlm-2_hyps.txt','w') as ff:
    for lines in f:
        if lines[0] == 'D':
            line = lines.split()
            for x in line:
                a.append(x)

word = a[0]
for x in a:
    if x == word:
        count2 += 1
    else:
        word = x
                
print(count2)


