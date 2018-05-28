""" Structured Classifier """
import numpy as np
from .linear_model import LinearModel

class StructuredClassifier:
    ''' An abstract structured classifier.'''

    def __init__(self, decoder):
        self.model = LinearModel()
        self.decoder = decoder
        self.regularization_constant = 1e12
        self.DEBUG = False
        self.initial_learning_rate = 0.001  # Only for training with SGD.
        # Only for training with SGD. Change to 'inv' for Pegasos-style
        # updating.
        self.learning_rate_schedule = 'invsqrt'
        self.use_pegasos_projection = False
        self.epoch = 0

    def train_svm_sgd(self, clusters_parts, clusters_features, num_epochs):
        '''Train an extractive summarizer with the svm-mira algorithm.'''
        for _ in range(num_epochs):
            self.train_svm_sgd_epoch(clusters_parts, clusters_features, self.epoch)
            self.epoch += 1

        return self.model

    def train_svm_sgd_epoch(self, clusters_parts, clusters_features, epoch):
        '''Run one epoch of the perceptron algorithm.'''
        total_loss = 0.0
        num_clusters = len(clusters_parts)
        lambda_coefficient = 1.0 / \
            (self.regularization_constant * num_clusters)
        t = num_clusters * epoch
        for i in range(num_clusters):
            parts = clusters_parts[i]
            features = clusters_features[i]
            scores = self.model.compute_scores(features)
            predicted_output, _, loss = self.decoder.decode_cost_augmented(parts, scores)

            if loss < 0.0:
                print('Negative loss: {}'.format(loss))
                loss = 0.0

            if loss is not np.nan:
                total_loss += loss
            num_parts = len(parts)
            assert len(predicted_output) == num_parts

            if self.learning_rate_schedule == 'invsqrt':
                eta = self.initial_learning_rate / np.sqrt(float(t + 1))
            elif self.learning_rate_schedule == 'inv':
                eta = self.initial_learning_rate / (float(t + 1))
            else:
                raise NotImplementedError

            decay = 1.0 - eta * lambda_coefficient
            assert decay >= -1e-12
            self.model.scale(decay)

            self.make_gradient_step(parts, features, eta, t, predicted_output)

            t += 1

        print('Epoch', epoch, 'Loss:', total_loss)

    def make_gradient_step(self, parts, features, eta, t, predicted_output):
        for i, part in enumerate(parts):
            if predicted_output[i] == part.gold:
                continue
            part_features = features[i]
            self.model.make_gradient_step(part_features, eta, t, predicted_output[i] - part.gold)