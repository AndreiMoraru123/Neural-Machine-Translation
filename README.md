# Attention Is All You Need

![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)  ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)  ![nVIDIA](https://img.shields.io/badge/nVIDIA-%2376B900.svg?style=for-the-badge&logo=nVIDIA&logoColor=white)


After training for a while (~60K steps), some interesting patterns arise. This project integrates them into TensorBoard's [Embedding Projector](https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin):

In the shared vocabulary between the encoder (english) and decoder (german) we can see some cosine similarities:

1. *Entschließung* (resolution) gets associated with *completed* 
![Resolution60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/2839ae4e-1cfd-4ca0-a160-fd1fd5abf948)

2. *gessammelt* (collected) gets associated with *decision* in english and *Bestimmung* (determination) as well as *verstärkt* (strenghtened)
![strengthened_collected_consistent60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/4c0743d5-3acd-4e95-a208-8f66e04d80ff)

3. *Change* also gets associated with *neuer* (new)
![change_neuer_60k](https://github.com/AndreiMoraru123/machine-translation/assets/81184255/7e4320a6-e543-4fc1-bccd-ad08683b38ae)
