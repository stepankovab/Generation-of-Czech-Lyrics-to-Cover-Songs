using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using LyricsGeneratorApp.Service;
using LyricsGeneratorApp.Models;
using LyricsGeneratorApp.Views.Shared;
using System;
using System.Collections.Generic;

namespace LyricsGeneratorApp.Controllers
{
    /// <summary>
    /// Controller for the lyrics generator.
    /// </summary>
    public class GeneratorController : Controller
    {
        private readonly LyricsGeneratorService lyricsGeneratorService;

        /// <summary>
        /// Create new GeneratorController
        /// </summary>
        public GeneratorController() 
        {
            lyricsGeneratorService = new();
        }

        /// <summary>
        /// Shows index.
        /// </summary>
        /// <returns>ViewResult.</returns>
        [HttpGet]
        public IActionResult Index()
        {
            return View();
        }

        /// <summary>
        /// Supplies generated lyrics with a custom structure. 
        /// </summary>
        /// <param name="DynamicTextBox">Contains info about the structure.</param>
        /// <param name="prompt">Prompt to continue with.</param>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="lyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <param name="syllables">Number of syllables on the line of the regenerated verse.</param>
        /// <param name="rhyme">Rhyme that will rhyme with the regenerated line.</param>
        /// <returns>ViewResult containing LyricsResponse containing lyrics with custom structure.</returns>
        [HttpPost]
        public IActionResult CustomLyrics(string[] DynamicTextBox, string prompt, string regenerateLyrics, string lyrics, int syllables, string rhyme)
        {
            LyricsResponse response;
            if (regenerateLyrics == null)
            {
                response = lyricsGeneratorService.GetCustomLyrics(DynamicTextBox, prompt);
            }
            else
            {
                response = lyricsGeneratorService.GetLyricsFix(regenerateLyrics, lyrics, syllables, rhyme);
            }

            return View(response);
        }

        /// <summary>
        /// Supplies generated sonet.
        /// </summary>
        /// <param name="prompt">Prompt to continue with.</param>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="lyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <returns>ViewResult containing LyricsResponse containing Sonet.</returns>
        [HttpPost]
        public IActionResult Sonet(string prompt, string regenerateLyrics, string lyrics)
        {
            LyricsResponse response;
            if (regenerateLyrics == null)
            {
                response = lyricsGeneratorService.GetSonet(prompt);
            }
            else
            {
                response = lyricsGeneratorService.GetLyricsFix(regenerateLyrics, lyrics, 0, null);
            }

            return View(response);
        }


        /// <summary>
        /// Supplies generated lyrics with the ABAB rhyme scheme.
        /// </summary>
        /// <param name="prompt">Prompt to continue with.</param>
        /// <param name="lines">Number of lines of the lyrics.</param>
        /// <param name="syllables">Number of syllables on the line of the regenerated verse.</param>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="lyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <param name="rhyme">Rhyme that will rhyme with the regenerated line.</param>
        /// <returns>ViewResult containing LyricsResponse containing lyrics with ABAB rhyme scheme.</returns>
        [HttpPost]
        public IActionResult ABABLyrics(string prompt, int lines, int syllables, string regenerateLyrics, string lyrics, string rhyme)
        {
            LyricsResponse response;
            if (regenerateLyrics == null)
            {
                response = lyricsGeneratorService.GetABABLyrics(prompt, syllables, lines);
            }
            else
            {
                response = lyricsGeneratorService.GetLyricsFix(regenerateLyrics, lyrics, syllables, rhyme);
            }

            return View(response);
        }


        /// <summary>
        /// Supplies generated lyrics with the AABB rhyme scheme.
        /// </summary>
        /// <param name="prompt">Prompt to continue with.</param>
        /// <param name="lines">Number of lines of the lyrics.</param>
        /// <param name="syllables">Number of syllables on the line of the regenerated verse.</param>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="lyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <param name="rhyme">Rhyme that will rhyme with the regenerated line.</param>
        /// <returns>ViewResult containing LyricsResponse containing lyrics with AABB rhyme scheme.</returns>
        [HttpPost]
        public IActionResult AABBLyrics(string prompt, int lines, int syllables, string regenerateLyrics, string lyrics, string rhyme)
        {
            LyricsResponse response;
            if (regenerateLyrics == null)
            {
                response = lyricsGeneratorService.GetAABBLyrics(prompt, syllables, lines);
            }
            else
            {
                response = lyricsGeneratorService.GetLyricsFix(regenerateLyrics, lyrics, syllables, rhyme);
            }

            return View(response);
        }

        /// <summary>
        /// Supplies new lyrics that should be song-compatible with the inputted original lyrics.
        /// </summary>
        /// <param name="originalLyrics">The original lyrics of which we take the structure.</param>
        /// <param name="regenerateLyrics">Verse to regenerate.</param>
        /// <param name="lyrics">The whole lyrics from which the verse is being regenerated.</param>
        /// <param name="syllables">Number of syllables on the line of the regenerated verse.</param>
        /// <param name="rhyme">Rhyme that will rhyme with the regenerated line.</param>
        /// <returns>ViewResult containing LyricsResponse containing rewritten lyrics.</returns>
        [HttpPost]
        public IActionResult RewriteLyrics(string originalLyrics, string regenerateLyrics, string lyrics, int syllables, string rhyme)
        {
            LyricsResponse response;
            if (regenerateLyrics == null)
            {
                response = lyricsGeneratorService.GetRewrittenLyrics(originalLyrics);
            }
            else
            {
                response = lyricsGeneratorService.GetLyricsFix(regenerateLyrics, lyrics, syllables, rhyme);
            }

            return View(response);
        }



        [Route("/error")]
        public IActionResult Error() => Problem();
    }
}
