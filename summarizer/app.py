#! /usr/bin/env python3

# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

import subprocess
import json

import requests
from flask import Flask, jsonify, make_response, request
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

import nltk

from priberam_summarizer.summarization_corpus import Document, Cluster, Sentence, Token, SummarizationCorpus
from priberam_summarizer.extractive_summarizer import CoverageExtractiveSummarizer, BasicCoverageExtractiveSummarizer
from priberam_summarizer.linear_model import LinearModel
from priberam_summarizer.vector_space_model import VectorSpaceModel

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


@app.route('/summarize', methods=['POST', 'GET'])
def summarize():
    
    model = request.args.get('model', 'media')
    if model == 'media':
        summarizer = cov_summarizer
        
    else:
        summarizer = basic_cov_summarizer

    word_count = int(request.args.get('maxWordCount', 50))
    summarizer.decoder.max_num_words = word_count

    corpus = SummarizationCorpus()
    cluster = Cluster()

    jdata = request.json
    for jdoc in jdata['documents']:

        document = Document()
        title = jdoc['instances'][0]['title']
        title = title.replace('\n' , ' ')
        sentence = Sentence(title)
        sentence.id = jdoc['id']
        setattr(sentence, 'doc_src_id', jdoc['id'])
        for word in nltk.word_tokenize(title):
            token = Token(word)
            setattr(token, 'lemma', word.lower())
            sentence.tokens.append(token)
        document.title.append(sentence)

        text = jdoc['instances'][0]['body']
        for line in nltk.sent_tokenize(text.replace('\n' , ' ')):
            sentence = Sentence(line)
            setattr(sentence, 'doc_src_id', jdoc['id'])
            for word in nltk.word_tokenize(line):
                token = Token(word)
                setattr(token, 'lemma', word.lower())
                sentence.tokens.append(token)
            document.body.append(sentence)

        cluster.documents.append(document)
    corpus.clusters.append(cluster)
    corpus.compute_tfidf()

    summary = summarizer.summarize(cluster)

    # template = {
    #     "highlights": [
    #         {
    #         "highlight": "string",
    #         "sentiment": {
    #             "value": "string",
    #             "mean": 0,
    #             "variance": 0
    #         },
    #         "language": "string",
    #         "sourceDocuments": [
    #             {
    #             "id": "string",
    #             "language": "string"
    #             }
    #         ]
    #         }
    #     ]
    #     }

    # summary is a Document()
    resp = {'highlights' : []}
    for sentence in summary.body:
        highlight = {
            "highlight": "string",
            "sentiment": {
                "value": "string",
                "mean": 0,
                "variance": 0
            },
            "language": "string",
            "sourceDocuments": [
                {
                "id": "string",
                "language": "string"
                }
            ]
            }
        highlight['highlight'] = sentence.text
        highlight['language'] = request.args.get('language', 'en')
        highlight['sourceDocuments'][0]['id'] = sentence.doc_src_id
        resp['highlights'].append(highlight)

    return jsonify(resp), 200

if __name__ == '__main__':
    print('Loading app...')

    # Model
    model = LinearModel()
    model.load('summarizer.model')

    # VectorSpace
    vector_space = VectorSpaceModel()
    vector_space.load_vector_space_model('vector_space.pickle')

    # Summarizer
    cov_summarizer = CoverageExtractiveSummarizer(vector_space, model=model, n_jobs=1)
    basic_cov_summarizer = BasicCoverageExtractiveSummarizer(vector_space, n_jobs=1).summarizer

    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(5000)
    IOLoop.current().start()

    #app.run(host='0.0.0.0', port= 5000)
