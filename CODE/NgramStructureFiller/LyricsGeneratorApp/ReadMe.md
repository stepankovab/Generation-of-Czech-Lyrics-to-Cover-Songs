# Overview of used algorithms

## Word n-gram language model

Word n-gram language model is a statistical language model, which gives distributed probabilities over possible following words, given the context.

The N of n-gram model specifies the size of the "window" considered while training and using the model.

For example a 3-gram model means that the model takes into account a sliding window of 3 words, where the first two are the context of the last word.

### Training of an n-gram model

The model is trained on large amounts of text, usually the larger the better. 

First, the text is tokenized and preprocessed. In our case, tokens signifying the start or end of a section are added in place of punctuation tokens.

Second, we slide an n-tokens wide window over the preprocessed text, saving context-word pairs into a dictionary, counting the appearances of these pairs.

In our case, for more failproof model, we also add (n-i)-grams for i in range(1, n-2), so the model has something to fall back on if it doesn't know the whole (n-1)-word context.

After counting all n-gram frequencies, the probabilities of continuations for each possible context are computed.

### Usage

The n-gram model can be used for getting distributed probabilities of what the next word should be based on already known context.

## Recursive line searcher

A randomized recursive searching algorithm, that stops when stop conditions are satisfied. 

A DFS through the n-gram model.

- The searcher starts either with some given context, or with a start symbol.
- The searcher chooses next word based on the probabilities of the n-gram model given the context. Possibilities are sorted by probability, and a threshold between 0 and 1 (with bigger probability to be closer to 0) is generated. The probabilities of possibilities are sumed up until they reach the threshold. The first word that reaches the threshold is added to "generated line".
    - If generated line satisfies end conditions (e.g. the number of syllables, specific rhyme at the end of the line...) the line is returned and algorithm ends.
    - If the conditions are not satisfied, but there is still possibility to satisfy them, the recursive line searcher is called again, with the "generated line" as context.
    - If the conditions are not satisfied and can not be satisfied by continuing with "generated line", last word is removed from the line and from possible continuations, probabilities are recomputed and recursive line searcher is called again.

There are fallbacks in case that the line searcher doesn't find anything.

First fallback is trying to search for the next word in the smaller model.

Another fallback is adding a stop symbol at the end of the context.

## Rhymer

Rhymer is made using regular expressions, and clustering together similar sounding word endings.

## Syllabator

Syllabator first masks a given word, identifying unseparable parts of the word, replacing characters by *V* for vocals, *C* for consonants, and *0* for unseparable characters. This is done by a series of regular expressions.

After that, another set of regular expressions divides the mask into syllables.

The last step is to map the split mask to the original word, and return the word divided into syllables.

## Generating lyrics

To generate lyrics, we first need to train the n-gram model on textual data.

Then, we need to input a structure the lyrics should follow, with specified syllable lengths and rhyme schemes.

After that, we call the recursive line searcher for each line, always giving the previous line as context.

All of the lines put together form lyrics with the given scheme.