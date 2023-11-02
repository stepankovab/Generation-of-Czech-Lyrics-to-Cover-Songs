# Generating song lyrics with a given structure, and researching of Czech-English lyrics translations

The goal of this year-project is to research the topic of automatic song translation with an emphasis on conserving the meaning, singability and structure, for further extension within the bachelor thesis.

This goal can be divided into two parts.

## Dataset research
For using more advanced techniques of lyrics translation as part of the bachelor thesis, we need to put together a dataset of lyrics and poems that were already translated from one language to another (in our case Czech and English). We will use web scraping techniques to obtain these data. In the `DATA` folder can be found all our findings so far.

We assume, that by the end of the year project, we will have a cleaned and formatted dataset of sufficient size, consisting mainly of translated songs from musicals and musical films. A subsection of the dataset will be translated poems.

## The naive approach
The naive approach will serve as a baseline, that is slightly better than just generating random words.

As part of the year project, we will develop an application which allows the user to input song lyrics in one language, and outputs song lyrics in the other language, with the same structure as the original lyrics (eg. the number of lines, syllables, rhyme schemas...). As this is the naive approach, the emphasis is mainly on the structure, as opposed to conserving the meaning.

### N-gram model
The main idea of the naive approach is based on a simple n-gram model, modified with constraints about the length of a line, or a rhyme ending of the line, as well as on the preferred words to use. These preferred words will be taken from the original lyrics inputted by the user, and translated to the target language.

Another option will be for the user to specify the structure and preferred words, and let the generator fill in this structure, not needing any input lyrics.

## Bachelor thesis expectations
The goal of the bachelor thesis is to develop a tool for translating musical songs between Czech and English.

By song translation, we mean translation of the song lyrics such that the core meaning of the song will be preserved, as well as the song structure, rhyme schemas and overall singability. Ideally, if a singer was provided with our translated lyrics, they should be able to sing it to the original melody naturally.

We want to try using different techniques, both statistical methods as well as the neural approach, and the finetuning of pretrained generative models.

It is not yet clear, whether the final tool will be made to generate full lyrics on its own, or if it will aid a human translator, by suggesting possible lines, each having different qualities, and it will be left to the translator to choose the best option for their own need. All of these questions should be answered by experiments done as part of the year project.
