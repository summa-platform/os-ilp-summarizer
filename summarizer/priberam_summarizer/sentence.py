# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

""" Sentence and ParsedSentence data strucutures """
import uuid

class Sentence:
    '''A sentence, containing untokenized and tokenized text.'''

    def __init__(self, text: str):
        self.id = uuid.uuid4()
        self.text = text
        self.tokens = list()

        self.position = -1
        self.is_gold = False

    def num_words(self) -> int:
        ''' Number of words of the sentence'''
        return len(self.text.split())

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text