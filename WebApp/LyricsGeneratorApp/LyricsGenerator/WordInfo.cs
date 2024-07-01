using System;

namespace LyricsGeneratorApp.LyricsGenerator
{

    /// <summary>
    /// Class containing information about a word in the Bayessian probability N-Gram model.
    /// </summary>
    public class WordInfo : IComparable<WordInfo>
    {
        /// <summary>
        /// The word of focus.
        /// </summary>
        public string Word;
        /// <summary>
        /// Number of occurrences of the word in all the text read by the N-Gram model.
        /// </summary>
        public int Count;
        /// <summary>
        /// Probability of choosing this word given this context. Computed as <code>(Sum of all possible continuations) / count</code>
        /// </summary>
        public double Probability;
        /// <summary>
        /// Context of the word, last N words concatenated to a string with space as a delimiter.
        /// Context can be shorter than N, so the model doesn't get lost when dealing with rare words.
        /// </summary>
        public string Context;

        /// <summary>
        /// Class constructor specifying word, count, probability and context.
        /// </summary>
        /// <param name="word">The word of focus.</param>
        /// <param name="count">Number of occurrences of the word in all the text read by the N-Gram model.</param>
        /// <param name="probability">Probability of choosing this word given this context. Computed as <code>(Sum of all possible continuations) / count</code></param>
        /// <param name="context">Context of the word, last N words concatenated to a string with space as a delimiter.
        ///                       Context can be shorter than N, so the model doesn't get lost when dealing with rare words.</param>
        public WordInfo(string word, int count, double probability, string context)
        {
            Word = word;
            Count = count;
            Probability = probability;
            Context = context;
        }

        public int CompareTo(WordInfo? other)
        {
            if (other == null) return 1;

            if (Probability < other.Probability) { return 1; }
            else if (Probability > other.Probability) { return -1; }
            else { return 0; }
        }
    }
}