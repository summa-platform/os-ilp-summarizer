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