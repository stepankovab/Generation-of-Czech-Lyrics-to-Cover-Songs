namespace LyricsGeneratorApp.Models
{
    /// <summary>
    /// Model
    /// </summary>
    public class LyricsResponse
    {
        /// <summary>
        /// Lyrics, each item of the list is one line.
        /// </summary>
        public List<string> Lyrics;

        /// <summary>
        /// Lyrics, joined into one string, separated by commas.
        /// </summary>
        public string LyricsAsString;
    }
}
