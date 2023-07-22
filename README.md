# Attention Is All You Need

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)

> [!NOTE]\
> I adapted the code from [this awesome PyTorch version](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/tree/master). Please check it out as well.

> [!IMPORTANT]\
> I am using python `3.9` with tensorflow `2.10` as this is their last available version for native-Windows on GPU.

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

#### 1. *Entschließung* (resolution) gets associated with *completed* 

![Resolution60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/2839ae4e-1cfd-4ca0-a160-fd1fd5abf948)

#### 2. *gessammelt* (collected) maps to *decision* and *Bestimmung* (determination) as well as *verstärkt* (strenghtened)

![strengthened_collected_consistent60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/4c0743d5-3acd-4e95-a208-8f66e04d80ff)

#### 3. *Change* also gets associated with *neuer* (new)

![change_neuer_60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/7e4320a6-e543-4fc1-bccd-ad08683b38ae)
