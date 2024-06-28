# Generation of Czech Lyrics to Cover Songs

This is the repository containing code for Thesis that explores the topic of generating Czech lyrics to English
cover songs. 

Songs are often adapted to different languages to make them more
available to people who do not necessarily speak the language of the original
song. During the translation process, however, it is essential to preserve the
singability of the text in relation to the melody of the original song, as well as the
meaning of the song, so that the translated text fits the context of the original.
Currently, such translations are done by hand. We analyze and present the first
approaches to solve this problem for Czech through automatic generation using
NLP methods. In our work, we create and provide a dataset consisting of pairs
of English song lyrics and their official Czech translations. We also provide a
dataset of pure Czech song lyrics. We compare the quality of several generative
language models. To thoroughly evaluate and analyze their quality, we introduce
several automatic metrics and take into account the results of manual evaluation.
We find that smaller trained models perform better than larger untrained models.
In addition, context is important for the generation of good covers. Finally, we
show that our task can be approached from both the translation and generation
point of view.

## Repo structure

- Data
    - Contains both the monolingual train dataset as well as the bilingual test dataset
- Scripts
    - Python scripts used both for training and testing the models.
    - To run inference, run the script `covermaker.py`.
    - Models are downloadable from [here](http://hdl.handle.net/11234/1-5507)

## Usage

Detailed usage is described in the text of the Thesis in chapter 7.