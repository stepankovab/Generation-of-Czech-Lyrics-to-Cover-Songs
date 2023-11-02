using System.Text;
using System.Text.RegularExpressions;

namespace LyricsGeneratorApp.LyricsGenerator
{
    public enum PromptWork
    {
        Use,
        DontUse,
        UseLastWord
    }

    public interface INGramModel
    {
        public string GenerateLineWithLength(string context, int lineSyllables, PromptWork continueWithPrompt);
        public string GenerateLineWithLengthAndRhyme(string context, string rhyme, int lineSyllables, PromptWork continueWithPrompt);
        public void ClearHistory();
    }

    /// <summary>
    /// Class containing the ngram model and methods related
    /// </summary>
    public class NGramModel : INGramModel
    {
        /// <summary>
        /// The maximum n of the ngrams.
        /// </summary>
        private int N { get; init; }

        /// <summary>
        /// Hash map of the model. <code> Key : Context</code><code> Value : Hash map of Possible continuations </code>
        /// </summary>
        private Dictionary<string, Dictionary<string, WordInfo>> model;

        /// <summary>
        /// Context queue. Memory of the model.
        /// </summary>
        private Queue<string> contextQueue;

        /// <summary>
        /// Rhymer implementing IRhymer interface.
        /// </summary>
        private IRhymer rhymer;

        /// <summary>
        /// Static symbol to signal natural pause in text.
        /// </summary>
        private static string endSymbol = "<end>";

        private bool log;



        #region Constructors

        /// <summary>
        /// Class constructor. <br/>
        /// Creates an empty N-Gram model with given parameters. <br/>
        /// Needs to be filled with text before usage, e.g.by public method:
        /// <code>
        /// ExtendByText( ... path ...)
        /// </code>
        /// </summary>
        /// <param name="N">Max number of word grams taken into consideration when both building and using the model.</param>
        /// <param name="strictnessOfRhymes">On a scale of 0 to 5, how strict are the rhyming rules. 5 is almost an identity and 0 is really benevolent.</param>
        public NGramModel(int N, int strictnessOfRhymes, bool log = false)
        {
            this.N = N;
            model = new Dictionary<string, Dictionary<string, WordInfo>>();
            contextQueue = new Queue<string>();
            rhymer = new Rhymer(strictnessOfRhymes);
            this.log = log;
        }

        /// <summary>
        /// Class constructor.<br/> 
        /// Creates an N-Gram model with given parameters.<br/>
        /// Fills the model with text from all files that can be found in the "path" folder.
        /// </summary>
        /// <param name="N">Max number of word grams taken into consideration when both building and using the model.</param>
        /// <param name="strictnessOfRhymes">On a scale of 0 to 5, how strict are the rhyming rules. 5 is almost an identity and 0 is really benevolent.</param>
        /// <param name="path">The path to the folder of the text files to learn from.</param>
        public NGramModel(int N, int strictnessOfRhymes, string path, bool log = false)
        {
            this.N = N;
            model = new Dictionary<string, Dictionary<string, WordInfo>>();
            contextQueue = new Queue<string>();
            rhymer = new Rhymer(strictnessOfRhymes);
            this.log = log;

            ExtendModelByFolderOfTexts(path);
        }

        #endregion



        #region Interface methods

        /// <summary>
        /// Generates a line with a given length in syllables. Clears the model history before generating, generates from the starting prompt given by the parameter context.
        /// </summary>
        /// <param name="context">Context from which the next line will be generated.</param>
        /// <param name="lineSyllables">Number of syllables on the line.</param>
        /// <param name="continueWithPrompt">Bool value if the prompt should be included in the final line.</param>
        /// <returns>The generated line as string.</returns>
        public string GenerateLineWithLength(string context, int lineSyllables, PromptWork continueWithPrompt)
        {
            prepareContext(context);

            // create line, add context if needed
            StringBuilder line = new StringBuilder();
            var syllablesNeeded = lineSyllables;

            if (continueWithPrompt == PromptWork.Use)
            {
                syllablesNeeded -= Syllabytor.CountSyllables(context);
                line = new StringBuilder(context + " ");
            }
            else if (continueWithPrompt == PromptWork.UseLastWord)
            {
                Regex rg = new Regex(@"\w+");
                var matches = rg.Matches(context);
                if (matches.Count > 0)
                {
                    syllablesNeeded -= Syllabytor.CountSyllables(matches[matches.Count - 1].Value);
                    line = new StringBuilder(matches[matches.Count - 1].Value + " ");
                }                
            }

            if (syllablesNeeded == 0)
            {
                return line.Remove(line.Length - 1, 1).ToString();
            }

            var myContext = new List<string>(contextQueue);

            // recursively search for the best solution.
            bool found = recursiveLineSearcher(myContext, null, syllablesNeeded, false, 0);

            // no solution is satisfactory of the request.
            if (!found)
            {
                return "    > model is too small to handle your request.\n";
            }

            return lineExtractor(syllablesNeeded, myContext, line);
        }



        /// <summary>
        /// Generate a line with a given length in syllables, ending with a word with the same rhyming scheme as parameter rhyme. Clears the model history before generating, generates from the starting prompt given by the parameter context.
        /// </summary>
        /// <param name="context">Context from which the next line will be generated.</param>
        /// <param name="rhyme">A word that should rhyme with the final word on the line.</param>
        /// <param name="lineSyllables">Number of syllables on the line.</param>
        /// <param name="continueWithPrompt">Boolean value if the prompt should be included in the final line.</param>
        /// <returns>The generated line as string.</returns>
        public string GenerateLineWithLengthAndRhyme(string context, string rhyme, int lineSyllables, PromptWork continueWithPrompt)
        {
            prepareContext(context);

            // create line, add context if needed
            StringBuilder rhymeLine = new StringBuilder();
            var syllablesNeeded = lineSyllables;

            if (continueWithPrompt == PromptWork.Use)
            {
                syllablesNeeded -= Syllabytor.CountSyllables(context);
                rhymeLine.Append(context);
                rhymeLine.Append(" ");
            }
            else if (continueWithPrompt == PromptWork.UseLastWord)
            {
                Regex rg = new Regex(@"\w+");
                var matches = rg.Matches(context);
                syllablesNeeded -= Syllabytor.CountSyllables(matches[matches.Count - 1].Value);
                rhymeLine.Append(matches[matches.Count - 1].Value);
                rhymeLine.Append(" ");
            }

            if (syllablesNeeded == 0)
            {
                if (rhymer.Rhymes(context, rhyme))
                {
                    return rhymeLine.Remove(rhymeLine.Length - 1, 1).ToString();
                }
                else
                {
                    return "    > Prompt has the correct length but wrong rhyme.\n";
                }

            }

            var myContext = new List<string>(contextQueue);
            // recursively search for the best solution.
            bool found = recursiveLineSearcher(myContext, rhyme, syllablesNeeded, false, 0);

            // There is no solution for this request, try to find at least a line with correct length.
            if (!found)
            {
                return GenerateLineWithLength(context, lineSyllables, continueWithPrompt);
            }

            return lineExtractor(syllablesNeeded, myContext, rhymeLine);
        }


        /// <summary>
        /// Clear the history of an N-Gram model.
        /// </summary>
        public void ClearHistory()
        {
            contextQueue.Clear();
        }

        #endregion



        #region Public methods

        /// <summary>
        /// Extends model by each text file in the given folder. The model learns all ngrams present in these texts. After going through all texts, the model recalculates probabilities of all words.
        /// </summary>
        /// <param name="path">Path to the folder.</param>
        /// <exception cref="DirectoryNotFoundException">Throws DirectoryNotFoundException if path does not exist.</exception>
        public void ExtendModelByFolderOfTexts(string path)
        {
            if (!Directory.Exists(path))
            {
                throw new DirectoryNotFoundException("The directory " + path + " does not exist.");
            }

            if (log) { Console.WriteLine("Adding text to the ngram model."); }

            string[] files = Directory.GetFiles(path);
            foreach (string fileName in files)
            {
                ExtendModelByText(fileName, false);
            }

            recalculateNGramProbabilities();
        }

        /// <summary>
        /// Extends model by the text of a file. The model learns all ngrams present in the text.
        /// </summary>
        /// <param name="path">Path to the text.</param>
        /// <param name="recalculateProbabilities">Recalculate probabilities of all words after uploading text. Unless calculated elsewhere, recalculateProbabilities must be set to TRUE.</param>
        /// <exception cref="FileNotFoundException">Throws FileNotFoundException if path does not exist.</exception>
        public void ExtendModelByText(string path, bool recalculateProbabilities)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException("The file " + path + " does not exist.");
            }

            StreamReader sr = new StreamReader(path);
            var fileContent = sr.ReadToEnd();

            // pad and clean the file
            string cleanedFileContent = "";
            for (int i = 0; i < N - 1; i++)
            {
                cleanedFileContent += endSymbol + " ";
            }
            cleanedFileContent += Regex.Replace(fileContent, "[.!?;,:\f]", " " + endSymbol + " ").ToLower();

            // find all matches (all words using czech alphabet + german äöü)
            Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+|" + endSymbol);
            MatchCollection matchedWords = rg.Matches(cleanedFileContent);
            var splitFileContent = new List<string>();
            foreach (Match m in matchedWords)
            {
                splitFileContent.Add(m.Value);
            }

            // add ngrams
            for (int i = 0; i < splitFileContent.Count - (N - 1); i++)
            {
                addNGrams(new List<string>(splitFileContent.GetRange(i, N)));
            }

            if (log) { Console.WriteLine("Added: " + path); }

            // recalculate probabilities
            if (recalculateProbabilities)
            {
                recalculateNGramProbabilities();
            }
        }




        #endregion



        #region Private methods

        /// <summary>
        /// Create all possible ngrams of the words ending with the last word. Add these ngrams to the model.
        /// </summary>
        /// <param name="words">N words to be added to the model as an ngram.</param>
        private void addNGrams(List<string> words)
        {
            string lastWord = words[words.Count - 1];

            // to eliminate one letter nonsence words
            var rg = new Regex(@"^[qwertypljhgfdxcbnměščřžýáíéúůóťďň]$");
            if (!rg.IsMatch(lastWord))
            {
                for (int contextLength = 1; contextLength < N; contextLength++)
                {
                    string context = string.Join(" ", words.GetRange(N - 1 - contextLength, contextLength));

                    // the context is already registered
                    if (model.ContainsKey(context))
                    {
                        // the continuation is already registered
                        if (model[context].ContainsKey(lastWord))
                        {
                            model[context][lastWord].Count += 1;
                        }
                        else
                        {
                            model[context].Add(lastWord, new WordInfo(lastWord, 1, 0.0, context));
                        }
                    }
                    else
                    {
                        model.Add(context, new Dictionary<string, WordInfo>());
                        model[context].Add(lastWord, new WordInfo(lastWord, 1, 0.0, context));
                    }
                }
            }
        }

        /// <summary>
        /// Recalculates ngram probabilities of all words in the model.
        /// </summary>
        private void recalculateNGramProbabilities()
        {
            foreach (var contextKey in model.Keys)
            {
                var continuationsSum = 0;
                foreach (var wordValue in model[contextKey].Values)
                {
                    continuationsSum += wordValue.Count;
                }

                foreach (var wordValue in model[contextKey].Values)
                {
                    wordValue.Probability = (double)wordValue.Count / continuationsSum;
                }
            }
            if (log) { Console.WriteLine("Probabilities recomputed."); }
        }

        /// <summary>
        /// Recalculates ngram probabilities of the given list of continuations.
        /// </summary>
        /// <param name="continuations">Continuations to recalculate the probabilities of.</param>
        private void recalculateContinuationsProbabilities(List<WordInfo> continuations)
        {
            var continuationsSum = 0;
            foreach (var continuation in continuations)
            {
                continuationsSum += continuation.Count;
            }
            foreach (var continuation in continuations)
            {
                continuation.Probability = (double)continuation.Count / continuationsSum;
            }
        }


        /// <summary>
        /// Prepare the model for generating by adding requested context to the context queue.
        /// </summary>
        /// <param name="context">Context to be added to the contextQueue</param>
        private void prepareContext(string context)
        {
            var cleanedPrompt = Regex.Replace(context, @"[.!?;,:]", " " + endSymbol + " ").ToLower();

            Regex rg = new Regex(@"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžäöü]+|" + endSymbol);
            MatchCollection matchedWords = rg.Matches(cleanedPrompt);

            foreach (Match m in matchedWords)
            {
                contextQueue.Enqueue(m.Value);
            }
        }


        /// <summary>
        /// Finds the right context and gets the continuations of that context sorted by probability.
        /// </summary>
        /// <returns>Sorted continuations of the correct context.</returns>
        private List<WordInfo>? getSortedContinuationsBasedOnContextQueue()
        {
            for (int i = N - 1; i > 0; i--)
            {
                while (contextQueue.Count > i)
                {
                    contextQueue.Dequeue();
                }
                if (model.ContainsKey(string.Join(" ", contextQueue)))
                {
                    // Create list of wordInfos
                    List<WordInfo> posibilitiesList = new();
                    foreach (var word in model[string.Join(" ", contextQueue)])
                    {
                        posibilitiesList.Add(word.Value);
                    }

                    // Sort the list
                    posibilitiesList.Sort();

                    return posibilitiesList;
                }
            }
            return null;
        }


        /// <summary>
        /// Recursive function searching for a line satisfying the constraints.
        /// </summary>
        /// <param name="context">Context from which the next word will be generated.</param>
        /// <param name="rhyme">A word that should rhyme with the final word on the line or null. If null, rhyme at the end of the line is ignored, searching only for the correct length.</param>
        /// <param name="syllablesNeeded">Number of syllables left to find.</param>
        /// <param name="found">Bool value if a satisfactory solution was found.</param>
        /// <param name="levelOfRecursion">Level of recursion.</param>
        /// <returns>Bool value if a satisfactory solution was found.</returns>
        private bool recursiveLineSearcher(List<string> context, string? rhyme, int syllablesNeeded, bool found, int levelOfRecursion)
        {
            context.ForEach(contextQueue.Enqueue);
            var possibleWordsSorted = getSortedContinuationsBasedOnContextQueue();
            var random = new Random();

            // the context has possible continuations
            if (possibleWordsSorted is not null)
            {
                // make a copy of continuations
                var unusedSortedContinuations = new List<WordInfo>(possibleWordsSorted);

                // iterate until there are no more options to choose from
                // or until a solution is found.
                while (unusedSortedContinuations.Any())
                {
                    string nextWord = "";
                    double threshold = random.NextDouble() * random.NextDouble();
                    var possibilitiesSum = 0.0;

                    // find a word to try out
                    foreach (var possibleWord in unusedSortedContinuations)
                    {
                        possibilitiesSum += possibleWord.Probability;
                        if (possibilitiesSum > threshold)
                        {
                            nextWord = possibleWord.Word;
                            unusedSortedContinuations.Remove(possibleWord);
                            recalculateContinuationsProbabilities(unusedSortedContinuations);
                            break;
                        }
                    }

                    // if endSymbol or one-letter word for the rhyme => skip
                    if (nextWord.Equals(endSymbol) || (syllablesNeeded == 1 && nextWord.Length <= 1))
                    {
                        continue;
                    }

                    // count syllables of possibility
                    var syllablesPossibility = Syllabytor.CountSyllables(nextWord);

                    if (syllablesPossibility == syllablesNeeded)
                    {
                        // possible solution is found
                        if (rhyme is null)
                        {
                            // just the length
                            context.Add(nextWord);
                            return true;

                        }
                        else
                        {
                            // check the rhyme
                            if (rhymer.Rhymes(rhyme, nextWord))
                            {
                                context.Add(nextWord);
                                return true;
                            }
                        }
                    }
                    else if (syllablesPossibility < syllablesNeeded)
                    {
                        // add the word
                        context.Add(nextWord);
                        syllablesNeeded -= syllablesPossibility;

                        // try to find the next one
                        found = recursiveLineSearcher(context, rhyme, syllablesNeeded, found, levelOfRecursion + 1);

                        if (found)
                        {
                            return found;
                        }

                        // remove current word from the context and move onto the next loop
                        context.RemoveAt(context.Count - 1);
                        context.ForEach(contextQueue.Enqueue);
                        syllablesNeeded += syllablesPossibility;
                    }
                }
            }

            // catch loop to avoid empty return lines as much as possible
            // adds more words into context, mainly endSymbols or the words from previous line
            var saveLoopIteration = 0;
            while (levelOfRecursion == 0 && !found && saveLoopIteration < N)
            {
                if (saveLoopIteration > N / 2 - 1)
                {
                    context.Add(endSymbol);
                }
                else if (context.Count > 2 && context[context.Count - 1].Equals(endSymbol))
                {
                    if (!context[context.Count - 2].Equals(endSymbol))
                    {
                        if (Syllabytor.CountSyllables(context[context.Count - 2]) < syllablesNeeded)
                        {
                            syllablesNeeded -= Syllabytor.CountSyllables(context[context.Count - 2]);
                        }
                    }
                    context.Add(context[context.Count - 2]);
                }
                else
                {
                    context.Add(endSymbol);
                }
                saveLoopIteration++;
                found = recursiveLineSearcher(context, rhyme, syllablesNeeded, found, levelOfRecursion + 1);
            }
            return found;
        }


        /// <summary>
        /// Extracts line from private context queue based on number of wanted syllables.
        /// </summary>
        /// <param name="syllablesNeeded">Number of syllables still needed to put on the line.</param>
        /// <param name="myContext">Context from which the line is extracted.</param>
        /// <param name="line">Stringbuilder, could contain context.</param>
        /// <returns>Line as string of words divided by a single space, ending with comma and linebreak.</returns>
        private string lineExtractor(int syllablesNeeded, List<string> myContext, StringBuilder line)
        {
            // extract solution from 'myContext' List
            var syllablesFound = 0;
            var wordsAccepted = 0;
            while (syllablesFound < syllablesNeeded)
            {
                var word = myContext[myContext.Count - wordsAccepted - 1];
                // ignore endSymbol when extracting line
                if (word.Equals(endSymbol))
                {
                    wordsAccepted++;
                    continue;
                }
                var wordLength = Syllabytor.CountSyllables(word);
                syllablesFound += wordLength;
                wordsAccepted++;
            }

            // create the line
            for (int i = myContext.Count - wordsAccepted; i < myContext.Count; i++)
            {
                var word = myContext[i];
                if (word.Equals(endSymbol))
                {
                    continue;
                }

                line.Append(word);
                line.Append(" ");
            }

            line.Remove(line.Length - 1, 1);
            line.Append(",\n");

            return line.ToString();
        }


        #endregion




    }
}