# Attention Is All You Need

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> [!NOTE]\
> I adapted the code from [this awesome PyTorch version](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/tree/master). Please check it out as well.

> [!IMPORTANT]\
> I am using python `3.9` with tensorflow `2.10` as this is their last available version for native-Windows on GPU.

## Steps
 1. `pip install -r requirements.txt`
 2. [download.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/download.py) downloads all the data (`en`-`de` file pairs from Europarl, Common Crawl and News Commentary) to the specified folder as argument.
 3. [encode.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/encode.py) filters the data based on the arguments (origin, maximum length etc.) and trains the BPE model, saving it to a file.
 4. [train.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/train.py) runs the whole training pipeline with top-down logic found in the file. Everything is managed by  the `Trainer` from [trainer.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/trainer.py) (logging embeddings, checkpointing etc.).
 5. [translate.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/translate.py) runs the model inference and optionally evaluates it with `sacrebleu` using the `Evaluator` from [evaluator.py](https://github.com/AndreiMoraru123/Neural-Machine-Translation/blob/main/evaluator.py).
 6. [docs](https://github.com/AndreiMoraru123/Neural-Machine-Translation/tree/main/docs) contains notes with svg drawings from [the original repo](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/tree/master) and markdown files explaining the choices I had to make for adaptating from one framework to another.

The code itself is heavily commented and you can get a feel for how language models work by looking at the [tests](https://github.com/AndreiMoraru123/Neural-Machine-Translation/tree/main/test).

## Overfitting on one sentence

Input sequence:

```
"I declare resumed the session of the European Parliament "
"adjourned on Friday 17 December 1999, and I would like "
"once again to wish you a happy new year in the hope that "
"you enjoyed a pleasant festive period."
```

Results in the following generated hypotheses (all should to be the top one and the exact label for this sentence):

Top generated sequence:
```
('Ich erkläre die am Freitag, dem 17. Dezember unterbrochene Sitzungsperiode '
 'des Europäischen Parlaments für wiederaufgenommen, wünsche Ihnen nochmals '
 'alles Gute zum Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.')
```
All generated sequences in the beam (k=5) search:
```
[{'hypothesis': 'Ich die am Freitag, dem 17. Dezember unterbrochene '
                'Sitzungsperiode des Europäischen Parlaments für '
                'wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum '
                'Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
  'score': -3.3601136207580566},
 {'hypothesis': 'Ich erkläre die am Freitag, dem 17. Dezember unterbrochene '
                'Sitzungsperiode des Europäischen Parlaments für '
                'wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum '
                'Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
  'score': -1.4448045492172241},
 {'hypothesis': 'Ich Ich erkläre die am Freitag, dem 17. Dezember '
                'unterbrochene Sitzungsperiode des Europäischen Parlaments für '
                'wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum '
                'Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
  'score': -3.1513545513153076},
 {'hypothesis': 'Ich erkläre die die am Freitag, dem 17. Dezember '
                'unterbrochene Sitzungsperiode des Europäischen Parlaments für '
                'wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum '
                'Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
  'score': -3.3080737590789795},
 {'hypothesis': 'Ich erkläre erkläre die am Freitag, dem 17. Dezember '
                'unterbrochene Sitzungsperiode des Europäischen Parlaments für '
                'wiederaufgenommen, wünsche Ihnen nochmals alles Gute zum '
                'Jahreswechsel und hoffe, daß Sie schöne Ferien hatten.',
  'score': -3.3361663818359375}]
```

These are negative as they are log probabilities, the closest to zero being the top sequence

As a sanity check, the BLUE score should be a perfect `100/100` in all cases:

```
INFO:root:13a tokenization, cased
INFO:root:BLEU = 100.00 100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.000 hyp_len = 34 ref_len = 34)
INFO:root:13a tokenization, caseless
INFO:root:BLEU = 100.00 100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.000 hyp_len = 34 ref_len = 34)
INFO:root:International tokenization, cased
INFO:root:BLEU = 100.00 100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.000 hyp_len = 34 ref_len = 34)
INFO:root:International tokenization, caseless
INFO:root:BLEU = 100.00 100.0/100.0/100.0/100.0 (BP = 1.000 ratio = 1.000 hyp_len = 34 ref_len = 34)
```

## Embeddings

After training for a while, some interesting patterns arise. This project integrates them into the [Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).

In the shared vocabulary between the encoder (english) and decoder (german) we can see some cosine similarities:

#### *Bedenken* (pondering) is closest to *glaube* (believe)
![image](https://github.com/AndreiMoraru123/Neural-Machine-Translation/assets/81184255/5de08f03-271c-498b-9ad8-5b49273e0a97)

#### *Entschließung* (resolution) gets associated with *completed* 

![Resolution60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/2839ae4e-1cfd-4ca0-a160-fd1fd5abf948)

#### *gessammelt* (collected) maps to *decision* and *Bestimmung* (determination) as well as *verstärkt* (strenghtened)

![strengthened_collected_consistent60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/4c0743d5-3acd-4e95-a208-8f66e04d80ff)

#### *Change* also gets associated with *neuer* (new)

![change_neuer_60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/7e4320a6-e543-4fc1-bccd-ad08683b38ae)
