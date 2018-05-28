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