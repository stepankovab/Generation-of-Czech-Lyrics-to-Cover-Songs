using System.Text.RegularExpressions;

namespace LyricsGeneratorApp.LyricsGenerator
{
    public interface IRhymer
    {
        public bool Rhymes(string w1, string w2);
    }


    /// <summary>
    /// Class containing methods for identifiing rhymes
    /// </summary>
    public class Rhymer : IRhymer
    {
        /// <summary>
        /// On a scale of 0 to 5, how strict are the rhyming rules.
        /// 5 is almost identity and 0 is really benevolent.
        /// </summary>
        public int Strictness { get; init; }

        /// <summary>
        /// Letters of word to consider while rhyming.
        /// </summary>
        private static int rhymeLength = 2;

        /// <summary>
        /// Class constructor specifying the strictness of the rhymes.
        /// </summary>
        /// <param name="strictnessOfRhymes">On a scale of 0 to 5, how strict are the rhyming rules.
        /// 5 is almost identity and 0 is really benevolent.</param>
        public Rhymer(int strictnessOfRhymes)
        {
            Strictness = strictnessOfRhymes;
        }





        /// <summary>
        /// Compares two words and returns TRUE if the words rhyme according to the Rhymer, and FALSE if they don't.
        /// </summary>
        /// <param name="word1">First word to compare.</param>
        /// <param name="word2">Second word to compare.</param>
        /// <returns></returns>
        public bool Rhymes(string word1, string word2)
        {
            var key1 = GetRhymeKey(word1);
            var key2 = GetRhymeKey(word2);
            return key1.Equals(key2);
        }





        /// <summary>
        /// Returns a key for RhymeKeeper.Rhymes hash map. <br/>
        /// The key returned depends on the strictness of the rhyme and length of the input word.
        /// </summary>
        /// <param name="word">The word to get the rhyme key of.</param>
        /// <returns>Rhyme Key of the input word.</returns>
        public string GetRhymeKey(string word)
        {
            word = word.ToLower();

            word = Regex.Replace(word, "ch", "h");

            // returns whole syllable
            if (Strictness >= 5)
            {
                var wordSyllables = Syllabytor.Syllabify(word);
                return wordSyllables[wordSyllables.Count - 1];
            }

            if (word.EndsWith(","))
            {
                word = word.Remove(word.Length - 1);
            }

            // create rhymeLength long ending
            var ending = "";
            if (word.Length < rhymeLength)
            {
                for (int i = 0; i < rhymeLength - word.Length; i++)
                {
                    ending += "*";
                }
                ending += word;
            }
            else
            {
                ending = word.Substring(word.Length - rhymeLength, rhymeLength);
            }

            if (Strictness < 5)
            {
                ending = Regex.Replace(ending, "[aá]", "A");
                ending = Regex.Replace(ending, "[eé]", "E");
                ending = Regex.Replace(ending, "([tdn])([yý])", "$1Y");
                ending = Regex.Replace(ending, "[iíyý]", "I");
                ending = Regex.Replace(ending, "[oó]", "O");
                ending = Regex.Replace(ending, "[uúů]", "U");
                ending = Regex.Replace(ending, "^.[mn]ě", "*Ně");

                // replaces vocal at the start of ending with * symbol
                if (rhymeLength > 2)
                {
                    ending = Regex.Replace(ending, "^[AEIOU]", "*");
                }
                
            }

            if (Strictness < 4)
            {
                // group together beginnings of words
                ending = Regex.Replace(ending, "^[sz]", "S");
                ending = Regex.Replace(ending, "^[dt]", "D");
                ending = Regex.Replace(ending, "^[gk]", "K");
                ending = Regex.Replace(ending, "^[bp]", "B");
                ending = Regex.Replace(ending, "^[vfw]", "V");
                ending = Regex.Replace(ending, "^[ďť]", "Ď");

                ending = Regex.Replace(ending, "[Dn]I", "ĎI");
            }

            if (Strictness < 3)
            {
                // replace beginnings of words
                ending = Regex.Replace(ending, "^[ščžřĎ]", "Š");
                ending = Regex.Replace(ending, "^[SDKBVrhjlcnm]", "*");

                // similar consonants
                ending = Regex.Replace(ending, "[sz]", "S");
                ending = Regex.Replace(ending, "[dt]", "D");
                ending = Regex.Replace(ending, "[gk]", "K");
                ending = Regex.Replace(ending, "[bp]", "B");
                ending = Regex.Replace(ending, "[vfw]", "V");
                ending = Regex.Replace(ending, "[ďť]", "Ď");
            }

            // shorten ending by one char
            if (Strictness < 2)
            {
                ending = ending.Substring(1);
            }

            // shorten ending by two chars
            if (Strictness < 1)
            {
                ending = ending.Substring(1);
            }

            return ending;
        }

    }
}