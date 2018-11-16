# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

""" Provide functions to read Priberam processed corpora for summarization """
import os
import re
import multiprocessing
from math import log, sqrt
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from collections import Counter

from lxml import etree

from .cluster import Cluster
from .document import Document
from .sentence import Sentence
from .token import Token
from .vector_space_model import VectorSpaceModel

import nltk


class SummarizationCorpus():
    ''' A corpus built over document clusters for automatic summarization systems'''
    def __init__(self, read_reference_summaries=True, n_jobs=-1):
        self.clusters = list()
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.language = 'en'
        self.read_reference_summaries = read_reference_summaries
        self.vector_space = None

#========================== Read Functions  ==========================#

    def read_corpus(self, path, max_folders=-1):
        ''' Interpret the folder as a preprocessed cluster corpus,
        where we have clusters as subfolder and then documents'''
        cluster_list = [cluster_name for cluster_name in os.listdir(path)
                        if os.path.isdir(os.path.join(path, cluster_name))]
        cluster_list.sort()
        if max_folders >= 0:
            cluster_list = cluster_list[:max_folders]
        if self.n_jobs == 1:
            for cluster_name in cluster_list:
                print('Processing cluster {} ...'.format(cluster_name))
                cluster_folder = os.path.join(path, cluster_name)
                cluster = self.read_cluster(cluster_folder)
                self.clusters.append(cluster)
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:

                futures = []
                for cluster_name in cluster_list:
                    cluster_folder = os.path.join(path, cluster_name)
                    futures.append(executor.submit(self.read_cluster, cluster_folder))

                for future in futures:
                    cluster = future.result()
                    #print('Finished to process cluster {} ...'.format(cluster.name))
                    self.clusters.append(cluster)

    def read_CNN_corpus(self, path, max_folders=-1):
        ''' Read CNN corpus '''
        document_list = [doc_name[:-12] for doc_name in os.listdir(os.path.join(path, 'docs')) if doc_name.endswith('.txt.turboparser')]

        document_list.sort()

        if max_folders >= 0:
            document_list = document_list[:max_folders+1]

        for index, doc_name in enumerate(document_list):

            cluster = Cluster()
            cluster.path = path
            cluster.name = doc_name
            doc_path = os.path.join(path, 'docs', doc_name)
            doc = self.read_document(doc_path)
            cluster.documents.append(doc)

            if self.read_reference_summaries:
                summary_path = os.path.join(path, 'models', doc_name[:-4] + '.summary')
                cluster.reference_summaries.append(self.read_document(summary_path))

            self.clusters.append(cluster)

            if index % 100 == 0:
                print('Read {}/{}'.format(index, len(document_list)))

    def read_cluster(self, path):
        ''' Read the document cluster from folder '''
        cluster = Cluster()
        cluster.path = path
        cluster.name = path.split('/')[-1]
        docs_path = os.path.join(path, 'docs')
        document_list = [doc_name for doc_name in os.listdir(docs_path)
                         if doc_name.endswith('.xml')]
        document_list.sort()
        for doc_name in document_list:
            doc_path = os.path.join(docs_path, doc_name)
            doc = self.read_document(doc_path)
            cluster.documents.append(doc)

        if self.read_reference_summaries:
            models_path = os.path.join(path, 'models')
            # models summaries (a.k.a. human summaries)
            models_list = [summary_name for summary_name in os.listdir(models_path)
                           if summary_name.endswith('.txt')]

            models_list.sort()

            for summary_file in models_list:
                summary_path = os.path.join(models_path, summary_file)
                cluster.reference_summaries.append(
                    self.read_document(summary_path))

        return cluster

    def read_document(self, path):
        ''' Read document txt or xml documents.'''
        document = Document()

        document.name = path.split('/')[-1][:-4]

        if path.endswith('.txt'):

            if False: #os.path.exists(path + '.turboparser'):
                with open(path + '.turboparser', 'r', encoding='utf8') as fp:
                    conll_text = fp.read()

                conll_grids = self.read_turboparser_output(conll_text)

                for i, grid in enumerate(conll_grids):
                    sentence = self.extract_sentence_conll(grid)
                    sentence.position = i
                    document.body.append(sentence)

            else:
                with open(path, 'r', encoding='utf8') as fp:
                    for line in fp:
                        line = line.strip()
                        if len(line) != 0:
                            sentence = Sentence(line)
                            for word in line.split(' '):
                                token = Token(word)
                                setattr(token, 'lemma', word.lower())
                                sentence.tokens.append(token)
                        document.body.append(sentence)

        elif path.endswith('.summary'):
            with open(path, 'r', encoding='utf8') as fp:
                for line in fp:
                    line = line.strip()
                    if len(line) != 0:
                        sentence = Sentence(line)
                        for word in line.split(' '):
                            token = Token(word)
                            setattr(token, 'lemma', word.lower())
                            sentence.tokens.append(token)
                    document.body.append(sentence)

        elif path.endswith('.xml'):
            # parse as TAC/DUC corpus

            root = etree.parse(path)

            if False: #os.path.exists(path + '.turboparser'):

                root = etree.parse(path + '.turboparser')

                conll_headline = self.read_turboparser_output(root.xpath('string(//HEADLINE)').strip())
                conll_grids = self.read_turboparser_output(root.xpath('string(//TEXT)').strip())

                # date is present in the document title. E.g. AFP_ENG_20050613.0282.xml
                prog = re.search(r'\d{8}', document.name)
                if prog:
                    try:
                        document.date = datetime.strptime(prog.group(), '%Y%m%d')
                    except ValueError:
                        print('Date not found for document: {}'.format(document.name))
                else:
                    print('Date not found for document: {}'.format(document.name))

                for i, grid in enumerate(conll_headline):
                    sentence = self.extract_sentence_conll(grid)
                    sentence.position = i
                    document.title.append(sentence)

                for i, grid in enumerate(conll_grids):
                    sentence = self.extract_sentence_conll(grid)
                    sentence.position = i
                    document.body.append(sentence)
            else:
                root = etree.parse(path)

                headline = root.xpath('string(//HEADLINE)').strip()                
                sentence = Sentence(headline)                
                for word in headline.split(' '):
                    token = Token(word)
                    setattr(token, 'lemma', word.lower())
                    sentence.tokens.append(token)
                document.title.append(sentence)

                text = root.xpath('string(//TEXT)').strip()
                for line in nltk.sent_tokenize(text.replace('\n' , ' ')):
                    sentence = Sentence(line)
                    for word in nltk.word_tokenize(line):
                        token = Token(word)
                        setattr(token, 'lemma', word.lower())
                        sentence.tokens.append(token)
                    document.body.append(sentence)                
        else:
            raise NotImplementedError

        return document

    def read_turboparser_output(self, conll_text):

        # clean blank spaces and extra \n in conll_sents
        conll_text = re.sub('\n[ \t]*\n[ \t\n]*', '\n\n', conll_text.strip())

        fields = ['id', 'word', 'lemma', 'pos', 'ner', 'coref', 'head', 'relation', 'predicate', 'args']

        conll_grids = [[{fields[i]: int(t) if i in [0, 6] else t
                         for i, t in enumerate(s.split('\t')[:9] + [s.split('\t')[9:]])}
                        for s in sent.split('\n') if len(s) >= 10]
                       for sent in conll_text.split('\n\n')]

        return conll_grids

    def extract_sentence_conll(self, grid):
        #sentence = Sentence(self.offset_detokenizer(grid))
        sentence = Sentence(' '.join([t['word'] for t in grid]))
        for line in grid:
            token = Token(line['word'])
            setattr(token, 'id', line['id'])
            setattr(token, 'lemma', line['lemma'])
            setattr(token, 'pos', line['pos'])
            setattr(token, 'ner', line['ner'])
            setattr(token, 'coref', line['coref'])
            setattr(token, 'head', line['head'])
            setattr(token, 'relation', line['relation'])
            setattr(token, 'predicate', line['predicate'])
            setattr(token, 'args', line['args'])
            sentence.tokens.append(token)
        return sentence

#===================== Compute N-Grams and TF-IDF =====================#

    def compute_tfidf(self, use_stemmer=True, lower_case=True):

        unigrams = Counter()
        bigrams = Counter()

        # Compute TF frequencies for unigrams and bigrams
        for cluster in self.clusters:
            setattr(cluster, 'unigrams', Counter())
            setattr(cluster, 'bigrams', Counter())
            for document in cluster.documents:
                setattr(document, 'unigrams', Counter())
                setattr(document, 'bigrams', Counter())
                for sentence in document.sentences:
                    setattr(sentence, 'unigrams', Counter())
                    setattr(sentence, 'bigrams', Counter())
                    tokens = [token.lemma if use_stemmer else token.word for token in sentence.tokens]
                    tokens = [token.lower() if lower_case else token for token in tokens]
                    sentence.unigrams = Counter(tokens)
                    tokens = ['__start__'] + tokens + ['__end__']
                    sentence.bigrams = Counter(zip(tokens, tokens[1:]))

                    # update the count in document
                    document.unigrams.update(sentence.unigrams)
                    document.bigrams.update(sentence.bigrams)

                # update the count in  cluster
                cluster.unigrams.update(document.unigrams)
                cluster.bigrams.update(document.bigrams)

            unigrams.update(cluster.unigrams)
            bigrams.update(cluster.bigrams)

        # Compute IDFs. Number of docs / number of documents the term happen
        unigram_idfs = Counter()
        bigram_idfs = Counter()
        for cluster in self.clusters:
            for document in cluster.documents:
                unigram_idfs.update({token: 1 for token in document.unigrams.keys()})
                bigram_idfs.update({token: 1 for token in document.bigrams.keys()})

        num_docs = sum([len(cluster.documents) for cluster in self.clusters])

        # inverse document frequency smooth
        unigram_idfs = {key: log(num_docs / (1 + value)) for key, value in unigram_idfs.items()}
        bigram_idfs = {key: log(num_docs / (1 + value)) for key, value in bigram_idfs.items()}

        unigram_idfs['__oov__'] = log(float(num_docs), 10)
        bigram_idfs['__oov__'] = log(float(num_docs), 10)

        for cluster in self.clusters:
            setattr(cluster, 'unigram_tfidf', Counter())
            setattr(cluster, 'bigram_tfidf', Counter())
            # compute tf_idf for unigrams
            cluster.unigram_tfidf = {key: freq * unigram_idfs[key] for key, freq in cluster.unigrams.items()}
            norm_factor = sqrt(sum([value * value for value in cluster.unigram_tfidf.values()]))
            cluster.unigram_tfidf = {key: value / norm_factor if norm_factor > 0 else value for key, value in cluster.unigram_tfidf.items()}

            # compute tf_idf for bigrams
            cluster.bigram_tfidf = {key: freq * bigram_idfs[key] for key, freq in cluster.bigrams.items()}
            norm_factor = sqrt(sum([value * value for value in cluster.bigram_tfidf.values()]))
            cluster.bigram_tfidf = {key: value / norm_factor if norm_factor > 0 else value for key, value in cluster.bigram_tfidf.items()}

            for document in cluster.documents:
                for sentence in document.sentences:
                    setattr(sentence, 'unigram_tfidf', Counter())
                    setattr(sentence, 'bigram_tfidf', Counter())
                    # compute tf_idf for unigrams
                    sentence.unigram_tfidf = {key: freq * unigram_idfs[key] for key, freq in sentence.unigrams.items()}
                    norm_factor = sqrt(sum([value * value for value in sentence.unigram_tfidf.values()]))
                    sentence.unigram_tfidf = {key: value / norm_factor if norm_factor > 0 else value for key, value in sentence.unigram_tfidf.items()}

                    # compute tf_idf for bigrams
                    sentence.bigram_tfidf = {key: freq * bigram_idfs[key] for key, freq in sentence.bigrams.items()}
                    norm_factor = sqrt(sum([value * value for value in sentence.bigram_tfidf.values()]))
                    sentence.bigram_tfidf = {key: value / norm_factor if norm_factor > 0 else value for key, value in sentence.bigram_tfidf.items()}

                    delattr(sentence, 'unigrams')
                    delattr(sentence, 'bigrams')
                delattr(document, 'unigrams')
                delattr(document, 'bigrams')
            delattr(cluster, 'unigrams')
            delattr(cluster, 'bigrams')

        self.vector_space = VectorSpaceModel(unigram_idfs, bigram_idfs)

#============================== Utilities =============================#

    def offset_detokenizer(self, grid):
        ''' Detokenize from turboparser output based on NLTK tokenizer using its offsets'''
        text = ''
        last_offset = -1
        for line in sorted(grid, key=lambda line: line['start_offset']):
            if last_offset != -1 and last_offset != line['start_offset']:
                text += ' '
            # NLTK convert quotes to PTB format
            text += line['word'].replace('``', '"').replace("''", '"')
            last_offset = line['end_offset'] + 1
        return text