# Text segmentation

We are using Hidden Markov Chain (HMC) for several text segmentation tasks : 

* Part-Of-Speech Tagging 
  * Dataset : Universal Dependencies English)
* Chunking
  * Dataset : CoNLL 2000
* Named-Entity-Recognition
  * Dataset : CoNLL 2003


We are implementing and comparing the score of the Viterbi algorithm and the Forward Backward algorithm.

# Results

| Accuracy  | Chunking           | NER | POS Tagging |
| :--------------- |:---------------:| :---------------:| -----:|
| Viterbi score  |   0.92        |  0.87 |0.93 |
| Forward Backward score | 0.92             |   0.85 |0.94 |