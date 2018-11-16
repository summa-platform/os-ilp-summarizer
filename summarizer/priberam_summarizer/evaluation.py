#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

"""
    Rouge Evalaution for Automatic Summarization

    This code is copyrighted to Priberam in the context of SUMMA project
"""

import os
import re
import shutil
import tempfile
from subprocess import getoutput

__author__ = "Pedro Paulo Balage"
__copyright__ = "Priberam, Summa Project"
__version__ = "0.1"
__maintainer__ = "Pedro Paulo Balage"
__email__ = "pedro.balage@priberam.pt"
__status__ = "Prototype"

ROUGE_PATH = '/opt/ROUGE/ROUGE-1.5.5.pl'
ROUGE_DATA = '/opt/ROUGE/data'

class ROUGE():

    def __init__(self):
        pass

    def rouge_output_to_dict(self, output):
        """
        Convert the ROUGE output into python dictionary for further processing. Extracted from pyrouge.
        """
        # 0 ROUGE-1 Average_R: 0.02632 (95%-conf.int. 0.02632 - 0.02632)
        pattern = re.compile(
            r"(\d+) (ROUGE-\S+) (Average_\w): (\d.\d+) "
            r"\(95%-conf.int. (\d.\d+) - (\d.\d+)\)")
        results = {}
        for line in output.split("\n"):
            match = pattern.match(line)
            if match:
                sys_id, rouge_type, measure, result, conf_begin, conf_end = \
                    match.groups()
                measure = {
                    'Average_R': 'recall',
                    'Average_P': 'precision',
                    'Average_F': 'f_score'
                }[measure]
                rouge_type = rouge_type.lower().replace("-", '_')
                key = "{}_{}".format(rouge_type, measure)
                results[key] = float(result)
                results["{}_cb".format(key)] = float(conf_begin)
                results["{}_ce".format(key)] = float(conf_end)
        return results


    def evaluate(self, model_summaries, system_summaries):
        """
            models_summaries = list of a list of reference summaries
            system_summaries = list of system summaries
        """

        system_dir = tempfile.mkdtemp()
        model_dir = tempfile.mkdtemp()
        config_dir = tempfile.mkdtemp()
        config_file = os.path.join(config_dir, 'rouge_config.xml')

        with open(config_file, 'w', encoding='utf-8') as fp_conf:
            fp_conf.write('<ROUGE-EVAL version="1.55">')
            for index, (models, system) in enumerate(zip(model_summaries, system_summaries)):
                model_elems = []
                for model_num, model in enumerate(models):
                    model_char = chr(65 + model_num)
                    model_name = 'gold.' + model_char + '.' + str(index) + '.txt'
                    model_path = os.path.join(model_dir, model_name)
                    model_elems.append("<M ID=\"{id}\">{name}</M>".format(id=model_char, name=model_name))
                    with open(model_path, 'w', encoding='utf8') as f:
                        f.write(model)
                system_name = 'predicted.' + str(index) + '.txt'
                system_path = os.path.join(system_dir, system_name)
                peer_elem = "<P ID=\"{id}\">{name}</P>".format(
                    id=1, name=system_name)
                with open(system_path, 'w', encoding='utf8') as f:
                    f.write(system)

                model_elems = "\n\t\t\t".join(model_elems)
                eval_string = """
        <EVAL ID="{task_id}">
            <MODEL-ROOT>{model_root}</MODEL-ROOT>
            <PEER-ROOT>{peer_root}</PEER-ROOT>
            <INPUT-FORMAT TYPE="SPL">
            </INPUT-FORMAT>
            <PEERS>
                {peer_elem}
            </PEERS>
            <MODELS>
                {model_elems}
            </MODELS>
        </EVAL>
                """.format(
                    task_id=index,
                    model_root=model_dir, model_elems=model_elems,
                    peer_root=system_dir, peer_elem=peer_elem)
                fp_conf.write(eval_string)
            fp_conf.write("</ROUGE-EVAL>")

        # Standard ROUGE Evaluation (retrieved from http://www-nlpir.nist.gov/projects/duc/duc2005/tasks.html)
        # /opt/ROUGE/ROUGE-1.5.5.pl -e /opt/ROUGE/data -n 2 -x -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -m rouge_config.xml

        # This is the orignal command for evaluation TAC 2008
        command = ROUGE_PATH + ' -e ' + ROUGE_DATA + ' -m -n 4 -w 1.2 -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -d ' + config_file

        output = getoutput(command)

        shutil.rmtree(system_dir)
        shutil.rmtree(model_dir)
        shutil.rmtree(config_dir)
        return self.rouge_output_to_dict(output)


def evaluate(clusters, summaries):
    model_summaries = []
    system_summaries = []
    for i, cluster in enumerate(clusters):
        system_summaries.append(summaries[i].get_text())

        models = []
        for summary in cluster.reference_summaries:
            models.append(summary.get_text())
        model_summaries.append(models)

        assert len(models) != 0

    rouge = ROUGE()
    metrics = rouge.evaluate(model_summaries, system_summaries)

    return metrics

def evaluate_against_oracle(oracles, summaries):
    model_summaries = []
    system_summaries = []

    for summary in summaries:
        system_summaries.append(summary.get_text())

    for summary in oracles:
        model_summaries.append([summary.get_text()])

    rouge = ROUGE()
    metrics = rouge.evaluate(model_summaries, system_summaries)

    return metrics