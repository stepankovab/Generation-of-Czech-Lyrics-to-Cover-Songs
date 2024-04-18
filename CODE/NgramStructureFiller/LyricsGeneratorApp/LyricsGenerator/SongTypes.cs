using System.Collections.Generic;
using System;
using System.Text.RegularExpressions;
using System.Diagnostics;

namespace LyricsGeneratorApp.LyricsGenerator
{
    /// <summary>
    /// Contains static methods using ngram model to generate differently structured songs.
    /// </summary>
    abstract class SongTypes
    {
        /// <summary>
        /// Creates an ngram model and writes a line according to the parameters.
        /// </summary>
        /// <param name="length">The length of the line.</param>
        /// <param name="rhyme">The rhyme of the line.</param>
        /// <param name="prompt">Prompt that precedes the line.</param>
        /// <returns>A line satisfying the given constraints.</returns>
        public static string RewriteLine(int length, string rhyme, string prompt)
        {

            Console.WriteLine("Rewriting a line...");
            Console.WriteLine();

            NGramModel model = new NGramModel(4, 3, Directory.GetCurrentDirectory() + Constants.PathToTexts);

            if (rhyme == null)
            {
                return model.GenerateLineWithLength(prompt, length, PromptWork.DontUse);
            }
            else
            {
                return model.GenerateLineWithLengthAndRhyme(prompt, rhyme, length, PromptWork.DontUse);
            }
        }

        /// <summary>
        /// Writes lyrics with a custom structure.
        /// </summary>
        /// <param name="structure">Each item of the list represents one line. The required form is: "*int number of syllables* *string symbol to represent a rhyme*".</param>
        /// <param name="prompt">Prompt to bein with.</param>
        /// <returns>Custom lyrics in a list, each item represents one line.</returns>
        public static List<string> WriteCustomLyrics(string[]? structure, string? prompt)
        {


            string fileName = @"C:/Users/barca/MOJE/BAKALARKA/CODE/LLMExperiments/pipeline.py";

            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(@"C:/Users/barca/AppData/Local/Programs/Python/Python311/python.exe", fileName)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            p.Start();

            string output = p.StandardError.ReadToEnd();
            p.WaitForExit();

            Console.WriteLine(output);

            List<string> result = new List<string> { output };
                        

            return result;
        }

        /// <summary>
        /// Writes lyrics with an ABAB rhyme scheme.
        /// </summary>
        /// <param name="prompt">Prompt to bein with</param>
        /// <param name="syllablesOnLine">Number of syllables per each line.</param>
        /// <param name="lines">Number of lines in the lyrics.</param>
        /// <returns>Lyrics with an ABAB rhyme scheme in a list, each item represents one line.</returns>
        public static List<string> WriteABABLyrics(string prompt, int syllablesOnLine, int lines)
        {
            if (prompt == null) { prompt = ""; }

            Console.WriteLine("Writing lyrics with AABB rhyme scheme...");
            Console.WriteLine();

            // init
            NGramModel model = new NGramModel(4, 3, Directory.GetCurrentDirectory() + Constants.PathToTexts);
            List<string> result = new();
            var line = "";
            var rhymeA = "";
            var rhymeB = "";

            // generate lines
            for (int i = 0; i < lines; i++)
            {
                if (i == 0)
                {
                    PromptWork promptWork = new();
                    if (Syllabytor.CountSyllables(prompt) <= syllablesOnLine)
                    {
                        promptWork = PromptWork.Use;
                    }
                    else
                    {
                        promptWork = PromptWork.DontUse;
                    }

                    line = model.GenerateLineWithLength(prompt, syllablesOnLine, promptWork);
                    result.Add(line);

                    // finding the last word to fix rhyme A
                    Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+");
                    MatchCollection matchedWords = rg.Matches(line);
                    rhymeA = matchedWords[matchedWords.Count - 1].Value;
                }
                else if (i == 1)
                {
                    line = model.GenerateLineWithLength(line, syllablesOnLine, PromptWork.DontUse);
                    result.Add(line);

                    // finding the last word to fix rhyme B
                    Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+");
                    MatchCollection matchedWords = rg.Matches(line);
                    rhymeB = matchedWords[matchedWords.Count - 1].Value;
                }
                else if (i % 2 == 1)
                {
                    line = model.GenerateLineWithLengthAndRhyme(line, rhymeA, syllablesOnLine, PromptWork.DontUse);
                    result.Add(line);
                }
                else
                {
                    line = model.GenerateLineWithLengthAndRhyme(line, rhymeB, syllablesOnLine, PromptWork.DontUse);
                    result.Add(line);
                }
            }

            return result;
        }

        /// <summary>
        /// Writes lyrics with an AABB rhyme scheme.
        /// </summary>
        /// <param name="prompt">Prompt to bein with</param>
        /// <param name="syllablesOnLine">Number of syllables per each line.</param>
        /// <param name="lines">Number of lines in the lyrics.</param>
        /// <returns>Lyrics with an AABB rhyme scheme in a list, each item represents one line.</returns>
        public static List<string> WriteAABBLyrics(string prompt, int syllablesOnLine, int lines)
        {
            if (prompt == null) { prompt = ""; }

            Console.WriteLine("Writing lyrics with AABB rhyme scheme...");
            Console.WriteLine();

            // init
            NGramModel model = new NGramModel(4, 3, Directory.GetCurrentDirectory() + Constants.PathToTexts);
            List<string> result = new();
            var line = "";

            // generate lines
            for (int i = 0; i < lines; i++)
            {
                if (i == 0)
                {
                    PromptWork promptWork = new();
                    if (Syllabytor.CountSyllables(prompt) <= syllablesOnLine)
                    {
                        promptWork = PromptWork.Use;
                    }
                    else
                    {
                        promptWork = PromptWork.DontUse;
                    }

                    line = model.GenerateLineWithLength(prompt, syllablesOnLine, promptWork);
                    result.Add(line);
                }
                else if (i % 2 == 1)
                {
                    // finding the last word to rhyme with
                    Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+");
                    MatchCollection matchedWords = rg.Matches(line);
                    var lastWord = matchedWords[matchedWords.Count - 1].Value;

                    line = model.GenerateLineWithLengthAndRhyme(line, lastWord, syllablesOnLine, PromptWork.DontUse);
                    result.Add(line);
                }
                else
                {
                    line = model.GenerateLineWithLength(line, syllablesOnLine, PromptWork.DontUse);
                    result.Add(line);
                }
            }

            return result;
        }

        /// <summary>
        /// Write a Sonet.
        /// </summary>
        /// <param name="prompt">Prompt to begin with.</param>
        /// <returns>Sonet as a list, each item represents one line.</returns>
        public static List<string> WriteSonet(string prompt)
        {

            Console.WriteLine("Writing a Sonet...");
            Console.WriteLine();

            if (prompt == null) { prompt = ""; }

            string[] structure = new string[14];

            structure[0] = "10 A";
            structure[1] = "10 B";
            structure[2] = "10 A";
            structure[3] = "10 B";
            structure[4] = "10 C";
            structure[5] = "10 D";
            structure[6] = "10 C";
            structure[7] = "10 D";
            structure[8] = "10 E";
            structure[9] = "10 F";
            structure[10] = "10 E";
            structure[11] = "10 F";
            structure[12] = "10 G";
            structure[13] = "10 G";

            return WriteCustomLyrics(structure, prompt);
        }






        public static List<string> GetRewrittenLyrics(string originalLyrics)
        {
            Console.WriteLine("Writing lyrics with a copied structure...");
            Console.WriteLine();

            List<string> listLyrics = new List<string>(originalLyrics.Split(','));
            Regex rg = new Regex(@"[\w]+");
            Random random = new();

            string[] structure = StructureExtractor.Extract(originalLyrics);
            string[] prompts = new string[listLyrics.Count];

            for (int i = 0; i < listLyrics.Count; i++)
            {
                var lyricLine = listLyrics[i];

                List<string> promptCandidates = new();

                foreach (Match m in rg.Matches(lyricLine))
                {
                    if (! Constants.Stopwords.Contains(m.Value))
                    {
                        promptCandidates.Add(m.Value);
                    }
                }

                prompts[i] = "";

                if (promptCandidates.Count > 1)
                {
                    prompts[i] = promptCandidates[random.Next(promptCandidates.Count)];
                }

            }

            // inits
            NGramModel model = new NGramModel(4, 3, Directory.GetCurrentDirectory() + Constants.PathToTexts);
            Dictionary<string, string> symbolToRhyme = new();
            List<string> result = new();
            string line = "";

            // iterate over each line of structure
            for (int i = 0; i < structure.Length; i++)
            {
                var lineStructure = structure[i];
                var prompt = "";
                if (i < prompts.Length)
                {
                    prompt = prompts[i];
                }

                PromptWork promptWork = new PromptWork();

                // empty line
                if (lineStructure == null)
                {
                    result.Add("");
                    line = "---";
                    continue;
                }

                // parse line structure
                MatchCollection matchedParts = rg.Matches(lineStructure);

                // empty line
                if (matchedParts.Count == 0)
                {
                    result.Add("");
                    line = "---";
                    continue;
                }

                // generate line if the line structure is correct (first is number of syllables, and has at most 2 parts)
                if (int.TryParse(matchedParts[0].Value, out int syllablesOnLine) && matchedParts.Count() < 3)
                {
                    if (Syllabytor.CountSyllables(prompt) <= syllablesOnLine)
                    {
                        promptWork = PromptWork.UseLastWord;
                    }
                    else
                    {
                        promptWork = PromptWork.DontUse;
                    }

                    // if there is specified rhyme schema and the rhyme is known
                    if (matchedParts.Count() == 2 && symbolToRhyme.ContainsKey(matchedParts[1].Value))
                    {
                        line = model.GenerateLineWithLengthAndRhyme(line + prompt, symbolToRhyme[matchedParts[1].Value], syllablesOnLine, promptWork);
                        result.Add(line);
                    }
                    // if the rhyme is unknown or there is no rhyme
                    else
                    {
                        line = model.GenerateLineWithLength(line + prompt, syllablesOnLine, promptWork);
                        result.Add(line);
                    }

                    // if there is specified rhyme schema and the rhyme is unknown, define the rhyme
                    if (matchedParts.Count() == 2 && !symbolToRhyme.ContainsKey(matchedParts[1].Value))
                    {
                        MatchCollection matchedWords = rg.Matches(line);
                        var lastWord = matchedWords[matchedWords.Count - 1].Value;
                        symbolToRhyme[matchedParts[1].Value] = lastWord;
                    }
                }
                // the structure is incorrect
                else
                {
                    line = "    > incorrect formatting: " + lineStructure;
                    result.Add(line);
                    line = "";
                }
            }

            return result;

        }
    }
}