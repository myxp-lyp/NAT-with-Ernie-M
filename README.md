# NAT-with-Ernie-M

This is the MSc program individual project from imperial college london. 

Author is Yupei Li

Easy instructions when you rerun the codes to get the results:

1. Attach the input file in .vscode dictionary. Find the related preprocessing, CAMLM training and DAD training command.
2. When find out the dataset, please refer to the original fairseq instruction. WMT14 en-de is generated from distilled data set.
3. It has to generated the correct filtered dict based on the data set. words_only.py gives an example.
4. Change related command in fairseq/nat_model.py and fairseq/model.py, find comment includes 'CAMLM'
5. Set break points when you want to test the loss and use CAMLM_loss.
6. Use command --CAMLM-load to set the CAMLM component you wish to replace
7. Use patience to avoid overfitting

8. Any questions email yl7622@ic.ac.uk
