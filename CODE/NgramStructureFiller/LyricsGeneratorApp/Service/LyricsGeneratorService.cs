using LyricsGeneratorApp.LyricsGenerator;
using LyricsGeneratorApp.Models;
using System.Linq;
using System.Text.RegularExpressions;

namespace LyricsGeneratorApp.Service
{
    /// <summary>
    /// Service containing functions to supply final lyrics to the Controller.
    /// </summary>
    public class LyricsGeneratorService
    {
        /// <summary>
        /// Supplies lyrics with a custom structure.
        /// </summary>
        /// <param name="structure">Structure of the lyrics.</param>
        /// <param name="prompt">Prompt to continue with</param>
        /// <returns>LyricsResponse containing custom lyrics.</returns>
        public LyricsResponse GetCustomLyrics(string[] structure, string prompt)
        {
            if (structure.Length == 0)
            {
                return formatLyricsResponse(new List<string> { "" });
            }

            var finalLyrics = SongTypes.WriteCustomLyrics(structure, prompt);
            return formatLyricsResponse(finalLyrics);
        }

        /// <summary>
        /// Changes one line of the supplied lyrics.
        /// </summary>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="allLyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <param name="syllables">Number of syllables on the line.</param>
        /// <param name="rhyme">Rhyme for the line.</param>
        /// <returns>LyricsResponse containing  the supplied lyrics with changed line.</returns>
        public LyricsResponse GetLyricsFix(string regenerateLyrics, string allLyrics, int syllables, string rhyme)
        {
            if (allLyrics is null || allLyrics == "")
            {

                var line = SongTypes.RewriteLine(syllables, rhyme, "");

                return formatLyricsResponse(new List<string> { line });
            }

            List<string> listLyrics = new List<string>(allLyrics.Split(','));
            string[] structure = StructureExtractor.Extract(allLyrics);

            Regex numberRg = new Regex(@"[0-9]+");
            MatchCollection matchedNumbers = numberRg.Matches(regenerateLyrics);


            if (matchedNumbers.Count == 0)
            {
                return new LyricsResponse
                {
                    Lyrics = listLyrics,
                    LyricsAsString = allLyrics,
                };
            }

            int lineFix = int.Parse(matchedNumbers[0].Value);

            // adjust from normal counting to computer indexing
            lineFix--;

            var context = "";
            if (lineFix > 0)
            {
                context = listLyrics[lineFix - 1];
            }

            var lineStructure = structure[lineFix];
            var splitLineStructure = lineStructure.Split(' ');

            if (splitLineStructure.Length == 2 && rhyme == null)
            {
                rhyme = StructureExtractor.GetRhymeKeyOfLine(listLyrics[lineFix]);
            }

            if (syllables == 0)
            {
                syllables = int.Parse(splitLineStructure[0]);
            }

            for (int i = 0; i < listLyrics.Count; i++)
            {
                listLyrics[i] += ",\n";
            }

            var newLine = SongTypes.RewriteLine(syllables, rhyme, context);
            listLyrics[lineFix] = newLine;

            return formatLyricsResponse(listLyrics);
        }

        /// <summary>
        /// Calls the extractor and custom lyrics generator on the extracted structure.
        /// </summary>
        /// <param name="originalLyrics">Lyrics which structure we want to replicate.</param>
        /// <returns>LyricsResponse containing  the supplied lyrics.</returns>
        public LyricsResponse GetRewrittenLyrics(string originalLyrics)
        {
            if (originalLyrics == null)
            {
                originalLyrics = "";
            }

            originalLyrics = originalLyrics.Replace('.', ',');


            while (! Regex.IsMatch(originalLyrics[originalLyrics.Length - 1].ToString(), @"\w"))
            {
                originalLyrics = originalLyrics.Remove(originalLyrics.Length - 1);
            }

            var newLyrics = SongTypes.GetRewrittenLyrics(originalLyrics);

            return formatLyricsResponse(newLyrics);
        }


        /// <summary>
        /// Calls the extractor and custom lyrics generator on the extracted structure.
        /// </summary>
        /// <param name="originalLyrics">Lyrics which structure we want to replicate.</param>
        /// <returns>LyricsResponse containing  the supplied lyrics.</returns>
        public LyricsResponse GetRewrittenLyricsGPT2(string originalLyrics)
        {
            if (originalLyrics == null)
            {
                originalLyrics = "";
            }

            originalLyrics = originalLyrics.Replace('.', ',');

            if (originalLyrics.Length == 0)
            {
                return formatLyricsResponse(new List<string> { "" });
            } 


            while (!Regex.IsMatch(originalLyrics[originalLyrics.Length - 1].ToString(), @"\w"))
            {
                originalLyrics = originalLyrics.Remove(originalLyrics.Length - 1);
            }

            var newLyrics = SongTypes.GetRewrittenLyricsGPT2(originalLyrics);

            return formatLyricsResponse(newLyrics);
        }



        /// <summary>
        /// Formats the lyrics into the LyricsResponse.
        /// </summary>
        /// <param name="finalLyrics">Lyrics to be formatted.</param>
        /// <returns>LyricsResponse.</returns>
        private LyricsResponse formatLyricsResponse(List<string> finalLyrics)
        {
            var finalStringLyrics = "";

            for (int i = 0; i < finalLyrics.Count; i++)
            {
                if (finalLyrics[i].Length > 2)
                {
                    finalStringLyrics += finalLyrics[i].Remove(finalLyrics[i].Length - 1);
                }

                if (i == 0 && finalLyrics[i].Length > 0)
                {
                    finalLyrics[i] = finalLyrics[i].First().ToString().ToUpper() + finalLyrics[i].Substring(1);
                }

                if (i == finalLyrics.Count - 1 && finalLyrics[i].Length > 0)
                {
                    finalLyrics[i] = finalLyrics[i].Remove(finalLyrics[i].Length - 2) + ".";
                }

                finalLyrics[i] = (i + 1).ToString() + ".\t\t>\t.\t.\t" + finalLyrics[i];
            }
            if (finalStringLyrics.Length > 0)
            {
                finalStringLyrics = finalStringLyrics.Remove(finalStringLyrics.Length - 1);
            }

            return new LyricsResponse
            {
                Lyrics = finalLyrics,
                LyricsAsString = finalStringLyrics,
            };
        }
    }
}
