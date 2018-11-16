# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from string import punctuation

from .structured_classifier import StructuredClassifier
from .document import Document
from .sentence import Sentence
from .structured_classifier import LinearModel

from .extractive_summarization_decoder import ExtractiveCoverageSummarizationDecoder
from .parts import SentencePart, ConceptPart

class BasicCoverageExtractiveSummarizer():

    def __init__(self, vector_space_model, max_num_words=100, min_sentence_length=3, max_sentence_length=100, coverage_min_sentences=2, n_jobs=-1):
        self.summarizer = CoverageExtractiveSummarizer(vector_space_model, max_num_words, min_sentence_length, max_sentence_length, coverage_min_sentences, None, n_jobs)
        model = LinearModel()
        for count in range(coverage_min_sentences, 40):
            fname = 'doc_count_{}'.format(count)
            model[fname] = float(count)
        self.summarizer.model = model

    def test(self, clusters):
        return self.summarizer.test(clusters)


class CoverageExtractiveSummarizer():
    ''' Abstract class for an extractive summarization system.'''

    def __init__(self, vector_space_model, max_num_words=100, min_sentence_length=3, max_sentence_length=100, coverage_min_sentences=2, model=None, n_jobs=-1):
        self.model = model
        self.decoder = ExtractiveCoverageSummarizationDecoder()
        self.decoder.max_num_words = max_num_words

        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.coverage_min_sentences = coverage_min_sentences

        self.structured_classifier = StructuredClassifier(self.decoder)
        self.structured_classifier.decoder.relax = False
        self.structured_classifier.learning_rate_schedule = 'inv'  # pegasos style
        self.structured_classifier.initial_learning_rate = 0.001

        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.vector_space_model = vector_space_model


    def train(self, clusters, oracles, num_epochs=10):
        assert len(clusters) == len(oracles)
        clusters_parts = [self.make_parts(cluster, oracle) for cluster, oracle in zip(clusters, oracles)]
        clusters_features = [self.make_features(cluster, parts) for cluster, parts in zip(clusters, clusters_parts)]
        self.model = self.structured_classifier.train_svm_sgd(clusters_parts, clusters_features, num_epochs)

        return self.model

    def test(self, clusters):
        '''Run the structured classifier on test data.'''
        summaries = []
        if self.n_jobs == 1:
            for cluster in clusters:
                summaries.append(self.summarize(cluster))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for cluster in clusters:
                    futures.append(executor.submit(self.summarize, cluster))

                for future in futures:
                    summaries.append(future.result())
        return summaries

    def summarize(self, cluster):
        # check if model was previous trained
        if self.model is None:
            print('Summarization model was not trained. Aborting')
            return None

        parts = self.make_parts(cluster)
        features = self.make_features(cluster, parts)
        scores = self.model.compute_scores(features)

        # Should use lpsolve or AD3?
        self.decoder.relax = True
        predicted_output = self.decoder.decode(parts, scores)

         # # Final summary and its score.
        summary = Document()
        #summary.name = 'Extractive Summary: ' + cluster.name
        summary.name = cluster.name
        for i, sentence in enumerate(cluster.sentences):
            if predicted_output[i] > 0.5:
                summary.body.append(sentence)

        return summary

    def make_parts(self, cluster, oracle=None):
        parts = list()

        if oracle is not None:
            make_gold = True
        else:
            make_gold = False

        #Make sentence parts.
        i = 0
        for document in cluster.documents:
            for sentence in document.sentences:
                part = SentencePart(sentence)
                if sentence.num_words() < self.min_sentence_length or sentence.num_words() > self.max_sentence_length:
                    part.active = False
                if make_gold:
                    part.gold = i in oracle.selected_sentences

                parts.append(part)
                i += 1

        concept_freqlist = {}
        concept_tfidfs = {}
        concept_sentences = {}
        concept_gold = {}
        sentence_index = 0
        for document in cluster.documents:
            for sentence in document.sentences:
                if sentence.position == 0:
                    weight = 2
                else:
                    weight = 1
                for concept in set(cluster.bigram_tfidf).intersection(set(sentence.bigram_tfidf)):
                    if ((concept[0] in self.vector_space_model.stopwords) and
                            (concept[1] in self.vector_space_model.stopwords)):
                        continue
                    if concept[0] in punctuation or concept[1] in punctuation:
                        continue
                    concept_freqlist[concept] = concept_freqlist.get(concept, 0) + weight
                    concept_tfidfs[concept] = concept_tfidfs.get(concept, []) + [sentence.bigram_tfidf[concept]]
                    concept_sentences[concept] = concept_sentences.get(concept, []) + [sentence_index]
                    if make_gold:
                        concept_gold[concept] = concept_gold.get(concept, False) or (sentence_index in oracle.selected_sentences)
                sentence_index += 1

        for concept in cluster.bigram_tfidf:
            if concept not in concept_freqlist:
                continue
            freq = concept_freqlist[concept]
            if freq < self.coverage_min_sentences:
                continue
            sentence_indices = concept_sentences[concept]
            part = ConceptPart(concept, freq, sentence_indices)
            part.tf_idf = sum(concept_tfidfs[concept]) / len(concept_tfidfs[concept])
            if make_gold:
                part.gold = concept_gold[concept]
            parts.append(part)

        return parts

    def make_features(self, cluster, parts):
        features = []

        for part in parts:
            if part.type == 'sentence':
                sentence_features = self.compute_sentence_features(cluster, part)
                features.append(sentence_features)
            elif part.type == 'concept':
                concept_features = self.compute_concept_features(cluster, part)
                features.append(concept_features)
            else:
                raise Exception('Unknown part type: ' + part.type)

        return features

    def compute_sentence_features(self, cluster, part):
        features = {}
        return features

    def compute_concept_features(self, cluster, part):
        features = {}

        use_counts_only = False
        use_all_bins = True
        use_bucketed_counts = True
        use_double_conjunctions = True
        use_triple_conjunctions = True
        use_bias = True


        concept = part.concept
        value = part.value
        sentence_indices = part.sentence_indices
        concept_type = 'bigram'
        # Only features with non-zero values need to be declared.

        # Don't create features for concepts with zero-value (e.g. which are
        # stopwords or don't occur frequently enough).
        if value <= 0.0:
            return features

        # killer_feature = cluster.name + ' ' + '_'.join(concept)
        # self[killer_feature] = 1.0

        # Feature set 0: bias feature.
        if use_bias:
            features['bias'] = 1.0

        # Feature set 1: how many documents in this cluster contain this concept.
        # Number of features = number of documents in the cluster.
        doc_count = int(value)

        score = part.tf_idf
        if 0 < score <= 0.25:
            features['score_0.25'] = 1
        elif 0.25 < score <= 0.5:
            features['score_0.5'] = 1
        elif 0.5 < score <= 0.75:
            features['score_0.75'] = 1
        elif 0.75 < score <= 1:
            features['score_1'] = 1
        elif score > 1:
            features['score_>1'] = 1

        if use_bucketed_counts:
            doc_count = int(math.floor(2 * math.log(2 + float(doc_count))))

        maximum_doc_count = 10000000  # 7

        if doc_count > maximum_doc_count:
            doc_count = maximum_doc_count

        if use_all_bins:
            for count in range(1, doc_count + 1):
                doc_count_feature_name = "doc_count_" + str(count)
                features[doc_count_feature_name] = 1.0
        else:
            doc_count_feature_name = "doc_count_" + str(doc_count)
            features[doc_count_feature_name] = 1.0

        # Feature set 2: which of the bigram's words have alphanumeric characters?
        # Number of features = 2 for unigrams, 4 for bigrams
        if not use_counts_only:
            if concept_type == 'unigram':
                if concept.lower() in self.vector_space_model.stopwords:
                    is_stop = "1"
                else:
                    is_stop = "0"
                is_stop_feature_name = "is_stop_" + is_stop
                features[is_stop_feature_name] = 1.0
            elif concept_type == 'bigram':
                if concept[0].lower() in self.vector_space_model.stopwords:
                    is_stop_1 = "1"
                else:
                    is_stop_1 = "0"

                if concept[1].lower() in self.vector_space_model.stopwords:
                    is_stop_2 = "1"
                else:
                    is_stop_2 = "0"
                is_stop_feature_name = "is_stop_" + is_stop_1 + is_stop_2
                features[is_stop_feature_name] = 1.0

            # Feature set 3: which is the earliest position of a sentence containing
            # this bigram in any document of the cluster?
            # This is binned into 0, 1, 2, 3+.
            # Number of features = 4
            earliest_position = 3
            for sentence_index in sentence_indices:
                sentence = cluster.sentences[sentence_index]
                if sentence.position < earliest_position:
                    earliest_position = sentence.position

            if earliest_position == 0:
                features['in_title'] = 1.0
            else:
                earliest_position_feature_name = "earliest_position_" + \
                str(earliest_position)
                features[earliest_position_feature_name] = 1.0


                # Feature set 4: two-way conjunctions of the above features
                # Number of features (for bigrams): N*4 + N*4 + 4*4 = 16 + 8*N, where
                # N is the number of documents in the cluster.
                if use_all_bins:
                    for count in range(1, doc_count + 1):
                        doc_count_feature_name = "doc_count_" + str(count)
                        if use_double_conjunctions:
                            features[doc_count_feature_name + "+" +
                                is_stop_feature_name] = 1.0
                            features[doc_count_feature_name + "+" +
                                earliest_position_feature_name] = 1.0

                        if use_triple_conjunctions:
                            # Feature set 5: three-way conjunctions of the above features
                            # Number of features (for bigrams): N*4*4 = 16*N
                            features[doc_count_feature_name + "+" + is_stop_feature_name +
                                "+" + earliest_position_feature_name] = 1.0
                else:
                    if use_double_conjunctions:
                        features[doc_count_feature_name + "+" +
                            is_stop_feature_name] = 1.0
                        features[doc_count_feature_name + "+" +
                            earliest_position_feature_name] = 1.0

                    if use_triple_conjunctions:
                        # Feature set 5: three-way conjunctions of the above features
                        # Number of features (for bigrams): N*4*4 = 16*N
                        features[doc_count_feature_name + "+" + is_stop_feature_name +
                            "+" + earliest_position_feature_name] = 1.0

                if use_double_conjunctions:
                    features[is_stop_feature_name + "+" +
                        earliest_position_feature_name] = 1.0

            # Total number of features (for bigrams): 24 + 25*N

            # Got worse results with these features
            if True:
                if concept_type == 'unigram':
                    if concept[0].isupper():
                        is_upper = "1"
                    else:
                        is_upper = "0"
                    is_upper_feature_name = "is_upper_" + is_upper
                elif concept_type == 'bigram':
                    if concept[0][0].isupper():
                        is_upper_1 = "1"
                    else:
                        is_upper_1 = "0"
                    if concept[1][0].isupper():
                        is_upper_2 = "1"
                    else:
                        is_upper_2 = "0"
                    is_upper_feature_name = "is_upper_" + is_upper_1 + is_upper_2
                features[is_upper_feature_name] = 1.0

        return features