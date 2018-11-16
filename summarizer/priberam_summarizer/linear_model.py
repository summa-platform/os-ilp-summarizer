# -*- coding: utf-8 -*-

# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

import math
import json

import numpy as np

class LinearModel(dict):
    ''' An abstract linear model.'''

    def compute_score(self, features):
        score = self.dot_product(features)
        return score

    def make_gradient_step(self, features, eta, t, gradient):
        self.add(features, -eta * gradient)

    def compute_scores(self, features):
        num_features = len(features)
        scores = np.zeros(num_features)
        for i in range(num_features):
            scores[i] = self.compute_score(features[i])
        return scores

    def save(self, filename):
        ''' Save a vector to a file path'''
        with open(filename, 'w', encoding='utf8') as fp:
            for key, value in sorted(self.items(), key=lambda t: t[1], reverse=True):
                fp.write('{}\t{}\n'.format(key, value))
            # json.dump(self.items(), fp, sort_keys=True, indent=4)

    def load(self, filename):
        ''' Load a vector from a file descriptor'''
        self.clear()        
        with open(filename, 'r', encoding='utf8') as fp:
            for line in fp:
                key, value = line.strip().split()
                self[key] = float(value)

    def add(self, vector, scalar=1.0):
        ''' Adds this vector and a given vector.'''
        for key in vector:
            self[key] = self.get(key, 0) + scalar * vector[key]

    def scale(self, scalar):
        '''Scales this vector by a scale factor.'''
        for key in self:
            self[key] *= scalar

    def add_constant(self, scalar):
        '''Adds a constant to each element of the vector.'''
        for key in self:
            self[key] += scalar

    def squared_norm(self):
        ''' Computes the squared norm, that is the product of the vector by itself. '''
        return self.dot_product(self)

    def dot_product(self, vector):
        ''' Computes the dot product with a given vector.'''
        return sum([self[key] * vector[key] for key in set(self).intersection(vector)])

    def normalize(self):
        ''' Normalize the vector. Note: if the norm is zero, do nothing.'''
        norm = math.sqrt(sum([value * value for value in self.values()]))
        if norm > 0.0:
            for key in self:
                self[key] /= norm