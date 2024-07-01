using System.Text.RegularExpressions;

namespace LyricsGeneratorApp.LyricsGenerator
{
    /// <summary>
    /// Class containing static methods for extracting the structure of lyrics.
    /// </summary>
    public class StructureExtractor
    {
        /// <summary>
        /// Analyze the line and return rhyme key according to Rhymer, of the last word of the line.
        /// </summary>
        /// <param name="line">Find the rhyme key of this line.</param>
        /// <returns>Rhyme key of the last word of the line.</returns>
        public static string GetRhymeKeyOfLine(string line)
        {
            Rhymer rhymer = new(3);

            Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+");
            MatchCollection matchedWords = rg.Matches(line.ToLower());

            var lastWord = "";
            if (line.Length > 0)
            {
                lastWord = matchedWords[matchedWords.Count - 1].Value;
            }

            return rhymer.GetRhymeKey(lastWord);
        }

        /// <summary>
        /// Extracts structure of the given lyrics according to the rules of Czech.
        /// </summary>
        /// <param name="lyrics">The lyrics which structure we want to extract.</param>
        /// <returns>Structure of the given lyrics.</returns>
        public static string[] Extract(string lyrics)
        {
            Dictionary<string, List<int>> rhymeToLine = new();

            var splitLyrics = lyrics.Split(',');

            string[] structure = new string[splitLyrics.Length];

            for (int i = 0; i < splitLyrics.Length; i++)
            {
                var line = splitLyrics[i];

                var rhymeKey = GetRhymeKeyOfLine(line);
                if (rhymeToLine.ContainsKey(rhymeKey))
                {
                    rhymeToLine[rhymeKey].Add(i);
                }
                else
                {
                    rhymeToLine[rhymeKey] = new List<int> { i };
                }

                structure[i] = Syllabytor.CountSyllables(line).ToString();
            }

            var symbol = "X";
            foreach (var rhymeKey in rhymeToLine.Keys)
            {
                if (rhymeToLine[rhymeKey].Count == 1)
                {
                    continue;
                }
                foreach (var lineNumber in rhymeToLine[rhymeKey])
                {
                    structure[lineNumber] += " " + symbol;
                }

                symbol += "X";
            }

            return structure;
        }

    }
}
