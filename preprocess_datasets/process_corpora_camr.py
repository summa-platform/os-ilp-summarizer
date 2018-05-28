#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Automaticaly process TAC corpora with CAMR parser

    This code is copyrighted to Priberam in the context of SUMMA project
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import json
import requests
import codecs
import time
from pprint import pprint

import multiprocessing as mp

__author__ = "Pedro Paulo Balage"
__copyright__ = "Priberam, Summa Project"
__version__ = "0.1"
__maintainer__ = "Pedro Paulo Balage"
__email__ = "pedro.balage@priberam.pt"
__status__ = "Prototype"


corpus_path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'datasets', 'TAC', 'TAC_preprocessed')


def request_camr(line):
    url = 'http://0.0.0.0:5000/api/parse'
    headers = {'Content-Type': 'text/plain',
               'Accept': 'application/json'}
    response = requests.post(
        url, headers=headers, data=line)
    resp = json.loads(response.content)[0]
    return resp

for dataset in os.listdir(corpus_path):
    dataset_path = os.path.join(corpus_path, dataset)
    for cluster in os.listdir(dataset_path):
        cluster_path = os.path.join(dataset_path, cluster)
        start = time.time()
        print(cluster_path)
        try:

            # read only process .body and .title
            filenames = [os.path.join(
                cluster_path, cluster + '.body'), os.path.join(cluster_path, cluster + '.title')]
            for filename in filenames:
                # if not os.path.exists(filename + '.camr.json'):
                lines = codecs.open(filename, encoding='utf8').readlines()
                lines = [line.strip() for line in lines]
                # process sentences in parallel
                pool = mp.Pool(1)
                amr_sentences = pool.map(request_camr, lines)
                pool.close()
                pool.join()
                # write .camr.json
                with codecs.open(filename + '.camr.json', 'w', encoding='utf8') as fp:
                    fp.write(json.dumps(amr_sentences))
        except:
            print('problem in {}'.format(cluster_path))

        end = time.time()
        print('Time spent: {} minutes'.format((end - start) / 60))
