import re


def dashed_syllabified_line(text : str):
    syll_line = syllabify(text, True)
    final_line = ""

    for syll in syll_line:
        if syll == "_":
            if len(final_line) > 0 and final_line[-1] == "-":
                final_line = final_line[:-1]
            final_line += " "
        else:
            final_line = final_line + syll + "-"
        
    if len(final_line) > 0 and final_line[-1] == "-":
        final_line = final_line[:-1]

    return final_line



def syllabify(text : str, dashed = False):

    words = re.findall(r"[aábcčdďeéěfghiíjklmnňoópqrřsštťuúůvwxyýzžAÁBCČDĎEÉĚFGHIÍJKLMNŇOÓPQRŘSŠTŤUÚŮVWXYÝZŽäöüÄÜÖ]+", text)
    syllables = []
    glued_preposition = False

    i = 0
    while i < len(words):
        word = words[i]

        glued_preposition = False
        if (word.lower() == "k" or word.lower() == "v" or word.lower() == "s" or word.lower() == "z") and i < len(words) - 1 and len(words[i + 1]) > 1:
            i += 1
            word = word + words[i]
            glued_preposition = True
        
        letter_counter = 0

        # Get syllables: mask the word and split the mask
        for syllable_mask in _split_mask(_create_word_mask(word)):
            word_syllable = ""
            for _ in syllable_mask:
                word_syllable += word[letter_counter]
                letter_counter += 1
                if dashed and glued_preposition and letter_counter == 1:
                    word_syllable += "_"

            syllables.append(word_syllable)

        if dashed:
            syllables.append("_")
        i += 1

    return syllables


  
def _create_word_mask(word : str) -> str:
    word = word.lower()

    vocals = r"[aeiyouáéěíýóůúäöü]"
    consonants = r"[bcčdďfghjklmnňpqrřsštťvwxzž]"

    replacements = [
        # osm
        ('osm', 'osu'),

        #double letters
		('ch', 'c0'),
		('rr', 'r0'),
        ('ll', 'l0'),
		('nn', 'n0'), 
		('th', 't0'),

        # au, ou, ai, oi
		(r'[ao]u', '0V'),
        (r'[ao]i','0V'),

        # eu at the beginning of the word
		(r'^eu', '0V'),
		
        # now all vocals
		(vocals, 'V'),

        # r,l that act like vocals in syllables
		(r'([^V])([rl])(0*[^0Vrl]|$)', r'\1V\3'),

        # sp, st, sk, št, Cř, Cl, Cr, Cv
		(r's[pt]', 's0'),
		(r'([^V0lr]0*)[řlrv]', r'\g<1>0'),
		(r'([^V0]0*)sk', r'\1s0'),
		(r'([^V0]0*)št', r'\1š0'),

		(consonants, 'K')
	]

    for (original, replacement) in replacements:
        word = re.sub(original, replacement, word)

    return word


def _split_mask(mask : str) -> list[str]:
    replacements = [
		# vocal at the beginning
		(r'(^0*V)(K0*V)', r'\1/\2'),
		(r'(^0*V0*K0*)K', r'\1/K'),

		# dividing the middle of the word
		(r'(K0*V(K0*$)?)', r'\1/'),
		(r'/(K0*)K', r'\1/K'),
		(r'/(0*V)(0*K0*V)', r'/\1/\2'),
		(r'/(0*V0*K0*)K', r'/\1/K'),

		# add the last consonant to the previous syllable
		(r'/(K0*)$', r'\1/')
	]

    for (original, replacement) in replacements:
        mask = re.sub(original, replacement, mask)

    if len(mask) > 0 and mask[-1] == "/":
        mask = mask[0:-1]

    return mask.split("/")