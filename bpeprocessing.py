import sentencepiece as spm

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.Load("/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori_newbpe/bpe.model")  # Replace with the actual path to your trained model
# Read input text
with open("/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori_newbpe/test1.de", "r", encoding="utf-8") as file:
    with open("/data/yl7622/NAT-with-Ernie-M/NAT_with_DAD-main/data-bin/wmt14_ende_distill_ori_newbpe/bpe/test.de", "w", encoding="utf-8") as f:

        text = file.readlines()
        
        for line in text:
            lines = line.split('\\n')
            for l in lines:
                tokens = sp.EncodeAsPieces(l)
                for token in tokens:
                    if token[-1] == '▁':
                        token = token[:-1]
                    else:
                        token += '@@'
                    f.write(token+ ' ')
                f.write(' \\n ')
            f.write('\n')
# Tokenize text using SentencePiece
#tokenized_text = sp.EncodeAsPieces(text)

# Write tokenized text to a new file

