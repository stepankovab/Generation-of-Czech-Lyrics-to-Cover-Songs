# LYRICS GENERATOR

LyricsGenerator is a web application that allows you to generate song lyrics.

In order to train use the generator, user needs to provide texts from which the ngram model in the lyrics generator will learn probabilities and continuations of words. Texts should be placed in the same folder as the `LyricsGeneratorApp.exe`.

The texts can be any Czech texts, preferably poems or song lyrics, books or newspaper articles.

<br>

## Usage

- Run the `LyricsGeneratorApp.exe`.

- Console application pops up with a message similar to this one: 

`Now listening on: http://localhost:5000`

- Copy and paste the `http://localhost` link into your favourite browser.

- Enjoy the web application!

<br>

## Types of supported schemes

- `Custom`
  - Uses the user-made rhyme scheme and line lengths.
- `ABAB`
  - Rhymes the odd lines together and the even lines together.
- `AABB`
  - Rhymes each pair of lines separately.
- `Sonet`
  - Uses the form of a Shakespearian Sonet. 14 lines of 10 syllables with the rhyming <i>ABAB CDCD EFEF GG</i>

<br>

## Example inputs and outputs

![Custom Lyrics Generator](img/Custom.png)

![Custom Lyrics Generator change line default](img/ChangeDefault.png)

![Custom Lyrics Generator change line](img/Change.png)

![AABB rhyme scheme lyrics generation](img/AABB.png)
