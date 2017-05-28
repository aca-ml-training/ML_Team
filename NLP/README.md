train

$ ./train.sh <training file> <model file name> 

test

$ ./test.sh <model file name> <test file>

I've tried seperating the entries containing the word "not" as well as separating the entries with the number of words >50 and >100 but it didn't give any difference in the result. I've tested only on 1.8ml instead of the whole dataset, so maybe it would make difference in the latter case. I also tried removing punctuation marks and nots but the accuracy became worse.
I haven't been able to test other things because I had some problems with my laptop and it worked really slow.. 