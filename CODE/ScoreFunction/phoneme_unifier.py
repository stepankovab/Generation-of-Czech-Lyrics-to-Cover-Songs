import eng_to_ipa as ipa
import re

def english_to_czech_pron(english):
    pronaunciation = ipa.convert(english)

    substitutions=[
        # solves 'every' problem
        ('vər', 'vr'),
        
        ('ˈ', ''),
        ('ɛ', 'e'),
        ('ə', 'e'),
        ('θ', 't'),
        ('ɔ', 'ou'),
        ('i', 'y'),
        ('ɪ', 'i'),
        ('u', 'ú'),
        ('ʊ', 'u'),
        ('ɑ', 'a'),
        ('ð', 'd'),
        ('æ', 'e'),
        ('ʃ', 'š'),
        ('ʒ', 'ž'),
        ('ʧ', 'č'),
        ('ʤ', 'ž'),
        ('ŋ', 'n'),
        ('ˌ', ''),
        ('oú', 'ou'),
        ('ks', 'x'),
        ('ts', 'c'),
        ('ouy', 'ouwi'),
        (r'([aeiou])\1', r'\1'),
        (r'([aeouáéóú])[yi]', r'\1j'),
        (r'([rj])l', r'\1')
    ]

    for (a,b) in substitutions: 
        pronaunciation=re.sub(a,b,pronaunciation)

    return pronaunciation

