import os 

file = "/data/yl7622/NAT-with-Ernie-M/data/wmt14_ende_distill/test.en"
ffile = "/data/yl7622/NAT-with-Ernie-M/data/wmt/test.en"

with open(file, 'r') as f:
    with open(ffile,'w') as ff:
        for line in f.readlines():
            s = ""
            for i in line.split():
                if i.isalpha() or i[-2:] == '@@':# only words
                #if not i.lower() == i.upper(): #make sure only words containing characters and could be labelled
                    s = s + i + '_@1@_ ' #__@1@__ for tgt token and __@0@__ for src token
                else:
                    s = s + i + ' '
                
            s = s.rstrip()
            
            ff.write(s + '\n')
            