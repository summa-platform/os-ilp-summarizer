""" Token in CONLL datastructure """

class Token:
    '''Token class keeps Token variables given by conll data structure
       Only word variable is required. All other are set dinamically using setattr
    '''

    def __init__(self, word: str):
        self.word = word

    def __str__(self):
        return self.word

    def __repr__(self):
        return self.word