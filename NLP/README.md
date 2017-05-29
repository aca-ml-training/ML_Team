I've tried seperating the entries containing the word "not" as well as separating the entries with the number of words >50 and >100 but it didn't give any difference in the result. 
I've tested only on 1.8ml instead of the whole dataset, so maybe it would make difference in the latter case. 
I also tried removing punctuation marks and nots but the accuracy became worse.
I haven't been able to test other things because I had some problems with my laptop and it worked really slow.. 
So I changed the parameters a little bit, I set the size of word vectors to 60, the minimal number of word accurances to 2
The length of character ngram to 4, the size of the window to 3 and the maximal length of the word ngrams to 3. 
