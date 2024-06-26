﻿using System.Text.RegularExpressions;
using System.Diagnostics;
using System.Text.Json;
using System.Text;
using Newtonsoft.Json;
using static System.Net.Mime.MediaTypeNames;

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

            var x = File.ReadAllText("config.json");
            (string _, string _, string _, string ngram_training_data_folder) = parseJsonResponse(x);

            NGramModel model = new NGramModel(4, 3, ngram_training_data_folder);

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

            Console.WriteLine("Writing lyrics with a custom scheme...");
            Console.WriteLine();

            var x = File.ReadAllText("config.json");
            (string _, string _, string _, string ngram_training_data_folder) = parseJsonResponse(x);

            // inits
            NGramModel model = new NGramModel(4, 3, ngram_training_data_folder);
            Dictionary<string, string> symbolToRhyme = new();
            List<string> result = new();
            string line = "";
            bool firstLine = true;

            // null safe
            if (prompt == null) { prompt = ""; }
            if (structure == null) { structure = Array.Empty<string>(); }

            // iterate over each line of structure
            foreach (string lineStructure in structure)
            {
                // empty line
                if (lineStructure == null)
                {
                    result.Add("");
                    line = "---";
                    continue;
                }

                // parse line structure
                Regex rg = new Regex(@"[\w]+");
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
                    // Continue with the prompt if it is short enough
                    if (firstLine)
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
                        firstLine = false;
                    }
                    // if there is specified rhyme schema and the rhyme is known
                    else if (matchedParts.Count() == 2 && symbolToRhyme.ContainsKey(matchedParts[1].Value))
                    {
                        line = model.GenerateLineWithLengthAndRhyme(line, symbolToRhyme[matchedParts[1].Value], syllablesOnLine, PromptWork.DontUse);
                        result.Add(line);
                    }
                    // if the rhyme is unknown or there is no rhyme
                    else
                    {
                        line = model.GenerateLineWithLength(line, syllablesOnLine, PromptWork.DontUse);
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
                prompts[i] = "";
            }

            var x = File.ReadAllText("config.json");
            (string _, string _, string _, string ngram_training_data_folder) = parseJsonResponse(x);

            // inits
            NGramModel model = new NGramModel(4, 3, ngram_training_data_folder);
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
        public static List<string> GetRewrittenLyricsGPT2(string originalLyrics)
        {
            Console.WriteLine("Writing lyrics with a copied structure...");
            Console.WriteLine();

            var x = File.ReadAllText("config.json");
            (string GPT2_model_folder, string python_exe, string python_scripts_folder, string _) = parseJsonResponse(x);



            string fileName = python_scripts_folder + "webapp_generate.py --input_section " + "\"" + originalLyrics + "\"" + " --model_path " + GPT2_model_folder;

            Process p = new Process();
            p.StartInfo = new ProcessStartInfo(python_exe, fileName)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                StandardOutputEncoding = Encoding.UTF8,
                StandardErrorEncoding = Encoding.UTF8,
            };
            p.Start();


            string output = p.StandardOutput.ReadToEnd();
            string err = p.StandardError.ReadToEnd();
            p.WaitForExit();

            Console.WriteLine(err);
            Console.WriteLine(output);

            byte[] byteOutput = Encoding.UTF8.GetBytes(output);

            var convOutput = Encoding.UTF8.GetString(byteOutput);

            List<string> result = new List<string>(convOutput.Split(','));

            return result;

        }

        private static (string, string, string, string) parseJsonResponse(string json_response)
        {
            // Parse Json response
            var acrDefinition = new
            {
                path_to_GPT2_model_folder = "",
                path_to_python_exe = "",
                path_to_python_scripts_folder = "",
                path_to_ngram_training_data_folder = ""
            };
            var acrObj = JsonConvert.DeserializeAnonymousType(json_response, acrDefinition);

            return (acrObj.path_to_GPT2_model_folder, acrObj.path_to_python_exe, acrObj.path_to_python_scripts_folder, acrObj.path_to_ngram_training_data_folder);
        }
    }
}