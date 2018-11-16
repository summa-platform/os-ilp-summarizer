# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

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