using System.Text;
using System.Text.RegularExpressions;

namespace LyricsGeneratorApp.LyricsGenerator
{

    /// <summary>
    /// Class containing static methods for Syllabifying Czech language.
    /// </summary>
    public class Syllabytor
    {
        /// <summary>
        /// Returns syllabified text as ArrayList of Strings. <br/>
        /// Only works for Czech language, trying to parse other languages will lead to incorrect results.
        /// </summary>
        /// <param name="text">Text to be syllabified.</param>
        /// <returns>List of syllables of the text.</returns>
        public static List<string> Syllabify(string text)
        {
            Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+");
            MatchCollection matchedWords = rg.Matches(text);

            var syllables = new List<string>();
            string word;

            for (int i = 0; i < matchedWords.Count; i++)
            {
                word = matchedWords[i].Value;

                if ((word.Equals("k") || word.Equals("v") || word.Equals("s") || word.Equals("z")) && i < matchedWords.Count - 1 && matchedWords[i + 1].Length > 1)
                {
                    i++;
                    word = word + matchedWords[i].Value;
                }

                var letterCounter = 0;
                // Get syllables: mask the word and split the mask
                foreach (var syllable in splitMask(createWordMask(word)))
                {
                    StringBuilder word_syllable = new StringBuilder();
                    foreach (var s in syllable)
                    {
                        word_syllable.Append(word[letterCounter++]);
                    }
                    syllables.Add(word_syllable.ToString());
                }
            }
            return syllables;
        }

        /// <summary>
        /// Returns number of syllables in the text. <br/>
        /// Only works for Czech language, trying to parse other languages will lead to incorrect results.
        /// </summary>
        /// <param name="text">Text to have syllables counted.</param>
        /// <returns>Number of syllables in text.</returns>
        public static int CountSyllables(string text)
        {
            if (text == null) return 0;

            Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+");
            MatchCollection matchedWords = rg.Matches(text);

            var syllableCounter = 0;
            string word;

            for (int i = 0; i < matchedWords.Count; i++)
            {
                word = matchedWords[i].Value;

                if (word.Equals("k") || word.Equals("v") || word.Equals("s") || word.Equals("z"))
                {
                    continue;
                }

                // Get syllables: mask the word and split the mask
                syllableCounter += splitMask(createWordMask(word)).Length;
            }

            return syllableCounter;
        }


        /// <summary>
        /// Returns a mask of the given word. <br/>
        /// Mask: <br/>
        /// C stands for consonant, V stands for vocal, 0 stands for ignore and count as previous syllable. <br/>
        /// This function implements part of an algorithm from an article from Naše řeč <a href="article">http://nase-rec.ujc.cas.cz/archiv.php?art=5348</a>
        /// </summary>
        /// <param name="word">The word to create a mask of.</param>
        /// <returns>Mask of the word.</returns>
        private static string createWordMask(string word)
        {
            word = word.ToLower();
            var vocals = "[aeiyouáéěíýóůú]";
            var consonants = "[bcčdďfghjklmnňpqrřsštťvwxzž]";

            // double letters
            word = Regex.Replace(word, "ch", "c0");
            word = Regex.Replace(word, "nn", "n0");
            word = Regex.Replace(word, "rr", "r0");
            word = Regex.Replace(word, "ll", "l0");
            word = Regex.Replace(word, "th", "t0");

            // au, ou, ai, oi
            word = Regex.Replace(word, @"[ao]u", "0V");
            word = Regex.Replace(word, @"[ao]i", "0V");

            // eu at the beginning of the word
            word = Regex.Replace(word, "^eu", "0V");

            // now all vocals
            word = Regex.Replace(word, vocals, "V");

            // r,l that act like vocals in syllables
            word = Regex.Replace(word, "([^V])([rl])(0*[^0Vrl]|$)", "$1V$3");

            // sp, st, sk, št, Cř, Cl, Cr, Cv
            word = Regex.Replace(word, "s[pt]", "s0");
            word = Regex.Replace(word, "([^V0lr]0*)([řlrv])", "$1X");

            // X was placeholder for 0 where regex thought I was writing $10 
            word = word.Replace('X', '0');

            word = Regex.Replace(word, "([^V0]0*)sk", "$1s0");
            word = Regex.Replace(word, "([^V0]0*)št", "$1š0");

            // all remaining consonants
            word = Regex.Replace(word, consonants, "C");

            return word;
        }

        /// <summary>
        /// Splits the mask made of vocals and consonants corresponding to original word into syllables. <br/>
        /// Mask: <br/>
        /// C stands for consonant, V stands for vocal, 0 stands for ignore and count as previous syllable. <br/>
        /// This function implements part of an algorithm from an article from Naše řeč <a href="article">http://nase-rec.ujc.cas.cz/archiv.php?art=5348</a>
        /// </summary>
        /// <param name="mask">The mask of the original word, see func. createMask()</param>
        /// <returns>String array of masked syllables.</returns>
        private static string[] splitMask(string mask)
        {
            // vocal at the beginning
            mask = Regex.Replace(mask, "(^0*V)(C0*V)", "$1/$2");
            mask = Regex.Replace(mask, "(^0*V0*C0*)C", "$1/C");

            // dividing the middle of the word
            mask = Regex.Replace(mask, "(C0*V(C0*$)?)", "$1/");
            mask = Regex.Replace(mask, "/(C0*)C", "$1/C");
            mask = Regex.Replace(mask, "/(0*V)(0*C0*V)", "/$1/$2");
            mask = Regex.Replace(mask, "/(0*V0*C0*)C", "/$1/C");

            // add the last consonant to the previous syllable
            mask = Regex.Replace(mask, "/(C0*)$", "$1/");

            if (mask.EndsWith("/"))
            {
                mask = mask.Remove(mask.Length - 1);
            }

            return mask.Split("/");
        }
    }
}