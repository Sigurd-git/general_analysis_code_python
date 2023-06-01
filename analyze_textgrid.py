import textgrid
def get_onset_offset_phones(textgrid):
    """
    Get onset and offset of each phone in a textgrid
    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset)
    """

    #get the intervals of phones from the textgrid
    names = [tier.name for tier in textgrid.tiers]
    phone_index = names.index('phones')
    intervals = textgrid.tiers[phone_index].intervals

    #get the onset and offset of each phone
    phones = []
    for interval in intervals:
        minTime, maxTime = interval.minTime, interval.maxTime
        phone = interval.mark
        if phone=='':
            continue
        
        #delete the number at the end of the phone
        if phone[-1].isdigit():
            phone = phone[:-1]

        phones.append((phone,minTime, maxTime))

    return phones
def get_onset_offset_words(textgrid):
    """
    Get onset and offset of each phone in a textgrid
    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset)
    """

    #get the intervals of phones from the textgrid
    names = [tier.name for tier in textgrid.tiers]
    phone_index = names.index('words')
    intervals = textgrid.tiers[phone_index].intervals

    #get the onset and offset of each phone
    words = []
    for interval in intervals:
        minTime, maxTime = interval.minTime, interval.maxTime
        word = interval.mark
        if word=='':
            continue

        words.append((word,minTime, maxTime))

    return words

def get_onset_offset_phones_words(textgrid):
    """
    Get onset and offset and the word it belongs to of each phone in a textgrid

    :param textgrid: TextGrid object
    :return: list of tuples of (phone, onset, offset, word, word onset, word offset,word index)
    """
    phonemes = get_onset_offset_phones(textgrid)
    words = get_onset_offset_words(textgrid)
    #get the onset and offset of each phone
    for i,phone in enumerate(phonemes):
        minTime, maxTime = phone[1], phone[2]
        for word in words:
            if word[1]<=minTime and word[2]>=maxTime:
                phonemes[i] = phonemes[i]+(word[0],word[1],word[2],words.index(word))

                # compute relative onset and offset of the phone
                phonemes[i] = phonemes[i]+(minTime-word[1],maxTime-word[1])
                break
        assert len(phonemes[i])==9, 'phoneme can not be aligned with word'
    return phonemes




if __name__ == '__main__':
    import glob
    import os
    import csv
    import scipy.io as sio
    import pickle
    textgrids = glob.glob('/scratch/snormanh_lab/shared/Sigurd/ljs/projects/dana/dana_aligned/*.TextGrid')
    for textgrid_file in textgrids:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
        # phones = [('phoneme','onset','offset')]+get_onset_offset_phones(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_phoneme.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(phones,f)
        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_phoneme.mat')
        # sio.savemat(mat_file,{'phones':phones})
        # words = [('word','onset','offset')]+get_onset_offset_words(tg)
        # #save as a pkl file
        # pkl_file = textgrid_file.replace('.TextGrid','_word.pkl')
        # with open(pkl_file,'wb') as f:
        #     pickle.dump(words,f)
        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
        
        phones_words = [('phoneme','phoneme_absolute_onset','phoneme_absolute_offset','word','word_absolute_onset','word_absolute_offset','word_index','phoneme_relative_onset','phoneme_relative_offset')]+get_onset_offset_phones_words(tg)
        #save as a pkl file
        pkl_file = textgrid_file.replace('.TextGrid','_phoneme_word.pkl')
        with open(pkl_file,'wb') as f:
            pickle.dump(phones_words,f)
        #save as a matlab file
        mat_file = textgrid_file.replace('.TextGrid','_phoneme_word.mat')
        sio.savemat(mat_file,{'phones_words':phones_words})
        #save as a csv file
        csv_file = textgrid_file.replace('.TextGrid','_phoneme_word.csv')
        with open(csv_file,'w') as f:
            writer = csv.writer(f)
            writer.writerows(phones_words)



    # for textgrid_file in textgrids:
    #     tg = textgrid.TextGrid.fromFile(textgrid_file)
    #     words = [('word','onset','offset')]+get_onset_offset_words(tg)

        #save as a matlab file
        # mat_file = textgrid_file.replace('.TextGrid','_word.mat')
        # sio.savemat(mat_file,{'words':words})
