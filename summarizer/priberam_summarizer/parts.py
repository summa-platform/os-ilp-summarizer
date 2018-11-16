# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

class Part():
    type = 'generic'
    item = None
    value = 0
    gold = False
    active = True

class SentencePart(Part):
    ''' Sentence part for structured prediction.'''
    def __init__(self, sentence):
        self.type = 'sentence'
        self.sentence = sentence

class ConceptPart(Part):
    ''' Concept part for structured prediction.'''
    def __init__(self, concept, frequency, sentence_indices):
        self.type = 'concept'
        self.concept = concept
        self.sentence_indices = sentence_indices  # Sentences in which this concept occurs.
        self.value = frequency
        self.tf_idf = 0