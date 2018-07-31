# A Deep Generative Framework for Paraphrase Generation

## Model:
This is the implementation of [A Deep Generative Framework for Paraphrase Generation](https://arxiv.org/pdf/1709.05074) by Ankush et al. (AAA2018) with Kim's [Character-Aware Neural Language Models](https://arxiv.org/abs/1508.06615) embedding for tokens. The code used the Samuel Bowman's [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349#) implementation as a base code available [here](https://github.com/kefirski/pytorch_RVAE).



## Usage
### Before model training it is necessary to train word embeddings for both questions and its paraphrases:
```
$ python train_word_embeddings.py --num-iterations 1200000
$ python train_word_embeddings_2.py --num-iterations 1200000
```

This script train word embeddings defined in [Mikolov et al. Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--num-sample` –– number of sampled from noise tokens


### To train model use:
```
$ python train.py --num-iterations 140000
```

#### Parameters:
`--use-cuda`

`--num-iterations`

`--batch-size`

`--learning-rate`

`--dropout` –– probability of units to be zeroed in decoder input

`--use-trained` –– use trained before model

### To sample data after training use:
```
$ python test.py
```
#### Parameters:
`--use-cuda`

`--num-sample`

