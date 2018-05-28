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