# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

""" A cluster of documents """
import uuid

class Cluster:
    '''A cluster of documents for summarization.'''

    def __init__(self):
        self.id = uuid.uuid4()
        self.name = ''
        self.path = ''
        self.documents = list()
        self.reference_summaries = list()
        self._index_sentences = None

    def get_sentence(self, sentence_id):
        if self._index_sentences is None:
            self._index_sentences = dict()
            # Build the Index
            for sentence in self.sentences:
                self._index_sentences[sentence.id] = sentence
        else:
            self._index_sentences.get(sentence_id, None)

    @property
    def sentences(self):
        ''' Return the title and body sentences for the documents for this cluster '''
        sentences = []

        for doc in self.documents:
            for sent in doc.sentences:
                sentences.append(sent)

        return sentences

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name