""" Provides Document data strucutre """
import uuid
from typing import List
from datetime import date

from .sentence import Sentence

class Document:
    '''A document containing a body and a title, which are lists of sentences.'''

    def __init__(self):
        self.id = uuid.uuid4()
        self.name = ''
        self.date = date.today()

        self.title = list()
        self.body = list()
        self.index_sentences = None

    def get_sentence(self, sentence_id):
        if self.index_sentences is None:
            self.index_sentences = dict()
            # Build the Index
            for sentence in self.sentences:
                self.index_sentences[sentence.id] = sentence
        else:
            self.index_sentences.get(sentence_id, None)

    @property
    def sentences(self) -> List[Sentence]:
        ''' Return the title and body sentences for the documents for this cluster '''
        sentences = []
        # for sent in self.title:
            # sentences.append(sent)

        for sent in self.body:
            sentences.append(sent)

        return sentences

    def num_sentences(self):
        ''' return the number of sentences '''
        return len(self.body)

    def get_text(self):
        text = '\n'.join([sentence.text for sentence in self.sentences])
        return text

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name