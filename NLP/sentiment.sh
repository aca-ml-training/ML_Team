./fasttext supervised -input train_file -output model_out_train -dim 60 -minCount 2 -maxn 4 -ws 3 -wordNgrams 3

./fasttext test model_out_train.bin test_file