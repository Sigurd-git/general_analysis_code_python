def attribute2phoneme(attrib, cmd=None, mode='Arpabet'):
    """
    Neural Acoustic Processing Lab,
    Columbia University, naplab.ee.columbia.edu
    """
    if cmd == 'list':
        phn = ['voiced', 'unvoiced', 'sonorant', 'syllabic', 'consonantal', 'approximant', 'plosive', 'fricative', 'nasal',
               'strident', 'labial', 'coronal', 'dorsal', 'anterior', 'front', 'back', 'high', 'low', 'obstruent']
        return phn

    if mode == 'Arpabet':
        phonemes = {
            'voiced': ['AA', 'AO', 'OW', 'AH', 'UH', 'UW', 'IY', 'IH', 'EY', 'EH', 'AE', 'AW', 'AY', 'OY', 'W', 'Y', 'L',
                       'R', 'M', 'N', 'NG', 'Z', 'V', 'DH', 'B', 'D', 'G', 'CH', 'JH', 'ER'],
            'unvoiced': ['TH', 'F', 'S', 'SH', 'P', 'T', 'K'],
            'sonorant': ['AA', 'AO', 'OW', 'AH', 'UH', 'UW', 'IY', 'IH', 'EY', 'EH', 'AE', 'AW', 'AY', 'OY', 'W', 'Y', 'L',
                         'R', 'M', 'N', 'NG'],
            'syllabic': ['AA', 'AO', 'OW', 'AH', 'UH', 'UW', 'IY', 'IH', 'EY', 'EH', 'AE', 'AW', 'AY', 'OY'],
            'consonantal': ['L', 'R', 'DH', 'TH', 'F', 'S', 'SH', 'Z', 'V', 'P', 'T', 'K', 'B', 'D', 'G', 'M', 'N', 'NG'],
            'approximant': ['W', 'Y', 'L', 'R'],
            'plosive': ['P', 'T', 'K', 'B', 'D', 'G'],
            'strident': ['Z', 'S', 'SH'],
            'labial': ['P', 'B', 'M', 'F', 'V'],
            'coronal': ['D', 'T', 'R', 'L', 'N', 'S', 'Z', 'SH'],
            'anterior': ['T', 'D', 'S', 'Z', 'TH', 'DH', 'P', 'B', 'F', 'V', 'M', 'N', 'L', 'R'],
            'dorsal': ['K', 'G', 'NG'],
            'front': ['IY', 'IH', 'EH', 'AE'],
            'back': ['UW', 'UH', 'AO', 'AA'],
            'high': ['IY', 'IH', 'UH', 'UW'],
            'low': ['EH', 'AE', 'AA', 'AO'],
            'nasal': ['M', 'N', 'NG'],
            'fricative': ['F', 'V', 'S', 'Z', 'SH', 'TH', 'DH'],
            'semivowel': ['W', 'L', 'R', 'Y'],
            'obstruent': ['DH', 'TH', 'F', 'S', 'SH', 'Z', 'V', 'P', 'T', 'K', 'B', 'D', 'G']
        }
    elif mode == 'IPA':
        phonemes = {
            'voiced': ['aa', 'ao', 'ow', 'axh', 'uxh', 'uw', 'iy', 'ixh', 'ey', 'eh', 'ae', 'aw', 'ay', 'oy', 'w', 'y', 'l',
                       'r', 'm', 'n', 'ng', 'z', 'v', 'dh', 'b', 'd', 'g', 'ch', 'jh', 'er'],
            'unvoiced': ['th', 'f', 's', 'sh', 'p', 't', 'k'],
            'sonorant': ['aa', 'ao', 'ow', 'axh', 'uxh', 'uw', 'iy', 'ixh', 'ey', 'eh', 'ae', 'aw', 'ay', 'oy', 'w', 'y', 'l',
                         'r', 'm', 'n', 'ng'],
            'syllabic': ['aa', 'ao', 'ow', 'axh', 'uxh', 'uw', 'iy', 'ixh', 'ey', 'eh', 'ae', 'aw', 'ay', 'oy'],
            'consonantal': ['l', 'r', 'dh', 'th', 'f', 's', 'sh', 'z', 'v', 'p', 't', 'k', 'b', 'd', 'g', 'm', 'n', 'ng'],
            'approximant': ['w', 'y', 'l', 'r'],
            'plosive': ['p', 't', 'k', 'b', 'd', 'g'],
            'strident': ['z', 's', 'sh'],
            'labial': ['p', 'b', 'm', 'f', 'v'],
            'coronal': ['d', 't', 'r', 'l', 'n', 's', 'z', 'sh'],
            'anterior': ['t', 'd', 's', 'z', 'th', 'dh', 'p', 'b', 'f', 'v', 'm', 'n', 'l', 'r'],
            'dorsal': ['k', 'g', 'ng'],
            'front': ['iy', 'ixh', 'eh', 'ae'],
            'back': ['uw', 'uxh', 'ao', 'aa'],
            'high': ['iy', 'ixh', 'uxh', 'uw'],
            'low': ['eh', 'ae', 'aa', 'ao'],
            'nasal': ['m', 'n', 'ng'],
            'fricative': ['f', 'v', 's', 'z', 'sh', 'th', 'dh'],
            'semivowel': ['w', 'l', 'r', 'y'],
            'obstruent': ['dh', 'th', 'f', 's', 'sh', 'z', 'v', 'p', 't', 'k', 'b', 'd', 'g']
        }

    return phonemes.get(attrib.lower(), [])


def phoneme2attribute(phn, cmd=None, mode=None):
    if mode is None or mode == '':
        mode = 'Arpabet'
    atr = []
    atlist = attribute2phoneme(phn,cmd='list', mode=mode)
    for cnt1 in range(len(atlist)):
        thisphn = attribute2phoneme(attrib=atlist[cnt1], mode=mode)
        if phn.lower() in [p.lower() for p in thisphn]:
            atr.append(atlist[cnt1])

    return atr

if __name__ == '__main__':
    phn = attribute2phoneme('plosive')
    phn = attribute2phoneme('plosive', mode='IPA')
    atr = phoneme2attribute('AA')
    atr = phoneme2attribute('axh', mode='IPA')
    pass



