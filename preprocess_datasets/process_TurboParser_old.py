#!/bin/env python3

import codecs
import os
import time
import urllib
from xml.parsers import expat
from xml.sax.saxutils import escape
import html
import nltk
from multiprocessing import Pool
import random

from turboparser import PyCppToPyTurboSink, PyCTurboTextAnalysis

import logging

global turbotextanalysis


# logging
logger = logging.getLogger()

logger.setLevel(logging.INFO)
handler = logging.FileHandler('turbo_parser.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Parser:

    def __init__(self):
        self._parser = expat.ParserCreate()
        self._parser.StartElementHandler = self.start
        self._parser.EndElementHandler = self.end
        self._parser.CharacterDataHandler = self.data
        self.process_tag = True
        self.unicode_xml = ''
        self.utf8_xml = ''

        self.sentences = []
        self.start_positions = []
        self.end_positions = []

        self.sentence_words = []
        self.sentence_start_positions = []
        self.sentence_end_positions = []
        self.sentences_start_positions = []
        self.sentences_end_positions = []

        self.debug = False

    def parse_xml(self, data):
        self.unicode_xml = data
        self.utf8_xml = data.encode('utf8')
        self._parser.Parse(data, 0)

    def close(self):
        self._parser.Parse("", 1)  # end of data
        del self._parser  # get rid of circular references

    def start(self, tag, attrs):
        # TAGS to not process
        exclude_tags = ['DATELINE']

        if tag in exclude_tags:
            self.process_tag = False

    def end(self, tag):
        self.process_tag = True
        self.autor = ''

    def data(self, data):
        start_offset = self._parser.CurrentByteIndex
        start_offset = len(self.utf8_xml[:start_offset].decode('utf8'))

        # expat by default expands html entities. Check if this is the case
        if (self.unicode_xml[start_offset] == '&' and
           (self.unicode_xml[start_offset + 4] == ';' or self.unicode_xml[start_offset + 3] == ';')):
            data = html.escape(data)

        end_offset = start_offset + len(data) - 1

        # check if the sentence is a continuation from previous data
        if len(self.end_positions) > 0 and start_offset == self.end_positions[-1] + 1:
            self.sentences[-1] += data
            self.end_positions[-1] = end_offset
        else:
            # dont save break lines
            if len(data.strip()) > 0:
                self.start_positions.append(start_offset)
                self.sentences.append(data)
                self.end_positions.append(end_offset)

    def process_sentences(self):
        for i, sentence in enumerate(self.sentences):
            start = 0
            for tok_sentence in nltk.sent_tokenize(sentence, language='english'):
                sentence_start = sentence.find(
                    tok_sentence, start) + self.start_positions[i]
                sentence_end = sentence_start + len(tok_sentence) - 1
                start = sentence.find(tok_sentence, start) + len(tok_sentence)

                tokens_start_positions = []
                tokens_end_positions = []
                index = 0
                tokens = []
                for token in nltk.word_tokenize(html.unescape(tok_sentence), 'english'):

                    # the problem of tokenization is that a token is transformed
                    # after the tokenization due to xml/html normalization
                    # here I am trying to guess the original form
                    # In case, of multiple in the text, I get the first
                    offsets = []
                    forms = []

                    forms.append(token)
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(token.replace('``', '"'))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(token.replace('``', "''"))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(token.replace("''", '"'))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(token.replace("/", '%2F'))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(urllib.parse.quote(token))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(urllib.parse.quote(token).lower())
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(urllib.parse.unquote(token))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(html.escape(token))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    forms.append(html.unescape(token))
                    offsets.append(tok_sentence.find(forms[-1], index))

                    # problem with %B1, encode to unknown, but unknown decode
                    # to '%EF%BF%BD'
                    if max(offsets) == -1 and tok_sentence[index] == '%' and index + 2 <= len(tok_sentence):
                        if urllib.parse.unquote(tok_sentence[index:index + 3]) == token:
                            forms.append(tok_sentence[index:index + 3])
                            offsets.append(tok_sentence.find(forms[-1], index))

                    if max(offsets) == -1:
                        # this should never happen. Am I missing something?
                        logger.error('Token offset not found')
                        return

                    # append the token modified by the tokenization
                    # Turbo will process this input
                    tokens.append(token)

                    # take the position and offsets based on original token
                    pos = min([offset for offset in offsets if offset != -1])
                    token = forms[offsets.index(pos)]
                    token_start_position = sentence_start + pos
                    token_end_position = sentence_start + pos + len(token) - 1
                    index = pos + len(token)
                    tokens_start_positions.append(token_start_position)
                    tokens_end_positions.append(token_end_position)
                    if self.debug:
                        print(token)
                        print(token_start_position)
                        print(token_end_position)
                        print(self.unicode_xml[
                              token_start_position:token_end_position + 1])
                        print()

                # dont save empty sentences and long chains
                if len(tokens_start_positions) != 0 and len(tokens) < 500:
                    self.sentence_words.append(tokens)
                    self.sentences_start_positions.append(
                        tokens_start_positions)
                    self.sentences_end_positions.append(tokens_end_positions)
                    self.sentence_start_positions.append(sentence_start)
                    self.sentence_end_positions.append(sentence_end)
        # self.check_sentences()

    def check_sentences(self):
        for i, sentence in enumerate(self.sentence_words):
            print('')
            print('--------------------')
            print(' '.join(sentence))
            print('-----------')
            start_offset = self.sentence_start_positions[i]
            end_offset = self.sentence_end_positions[i]
            print(start_offset, end_offset)
            print(self.sentences_start_positions[i])
            print(self.sentences_end_positions[i])
            print(self.unicode_xml[start_offset:end_offset + 1])
            print('--------------------')

    def check_extraction(self):
        for i, sentence in enumerate(self.sentences):
            print('')
            print('--------------------')
            print(sentence)
            print('-----------')
            start_offset = self.start_positions[i]
            end_offset = self.end_positions[i]
            print(self.unicode_xml[start_offset:end_offset + 1])
            print('--------------------')


def save_annotation(xml_text, tokens_info, sentences_start_positions, sentences_end_positions, sentence_start_positions, sentence_end_positions):

    # build the grids by offset

    grids = []
    grid_begin_positions = []
    grid_end_positions = []

    index = 0
    for i in range(len(sentence_start_positions)):
        grid = []
        for j, x in enumerate(tokens_info[index:index + len(sentences_start_positions[i])]):
            start_offset = sentences_start_positions[i][j]
            end_offset = sentences_end_positions[i][j]
            word = x['word']
            coref = x['features']['coref_info']
            head = x['features']['dependency_head']
            relation = x['features']['dependency_relation']
            entity = x['features']['entity_tag']
            id = x['features']['id']
            lemma = x['features']['lemma']
            tag = x['features']['pos_tag']
            pred = x['features']['semantic_predicate']
            args = x['features']['semantic_arguments_list'].split('|')

            # word and lemma should be transformed to avoid xml problems
            word = escape(word)
            lemma = escape(lemma)

            grid.append('\t'.join([str(item) for item in [
                        start_offset, end_offset, id, word, lemma, tag, entity, coref, head, relation, pred]] + args))

        grids.append('\n'.join(grid))
        grid_begin_positions.append(sentence_start_positions[i])
        grid_end_positions.append(sentence_end_positions[i])

        index = index + len(sentences_start_positions[i])

    assert len(grids) == len(grid_end_positions) == len(grid_begin_positions)

    # save the grid in the xml_text, changing the text they represent from
    # bottom to up
    for begin_position, end_position, grid in sorted(list(zip(grid_begin_positions, grid_end_positions, grids)), reverse=True):
        xml_text = xml_text[:begin_position] + '\n' + \
            grid + '\n' + xml_text[end_position + 1:]

    # exeption expat escape and it is not processed
    xml_text = xml_text.replace('&#10;', '').replace('&#13;', '')
    xml_text = xml_text.replace('#10;', '').replace('#13;', '')

    return xml_text


def process_xml(xml_text):

    global turbotextanalysis

    start_time = time.time()

    p = Parser()
    p.parse_xml(xml_text)
    p.process_sentences()

    sentences_words = p.sentence_words
    sentence_start_positions = p.sentence_start_positions
    sentence_end_positions = p.sentence_end_positions
    sentences_start_positions = p.sentences_start_positions
    sentences_end_positions = p.sentences_end_positions

    p.close()

    try:

        # Process the text with Turbo
        sink = PyCppToPyTurboSink(True)
        retval = turbotextanalysis.analyse_with_tokens('en',
                                                       sentences_words,
                                                       sentence_start_positions,
                                                       sentences_start_positions,
                                                       sentences_end_positions, sink)

        if retval != 0:
            logger.error("ERROR in PyCTurboTextAnalysis analyse")
            logger.error("Return value: {}".format(retval))
            exit()

        tokens_info = sink.get_tokens_info()

        del sink

    except Exception as exp:
        logger.error(
            'Turbo Parser Exception. {}'.format(exp))
        exit()
        return None

    if tokens_info is None or len(tokens_info) == 0:
        return

    xml_output = save_annotation(xml_text, tokens_info, sentences_start_positions, sentences_end_positions, sentence_start_positions, sentence_end_positions)

    end_time = round(time.time() - start_time)
    (t_min, t_sec) = divmod(end_time, 60)
    logger.info('Processing time: {} min {} sec'.format(t_min, t_sec))

    return xml_output

def process_txt(text):

    global turbotextanalysis

    start_time = time.time()

    p = Parser()
    p.start_positions = [0]
    p.sentences = [text]
    p.process_sentences()

    sentences_words = p.sentence_words
    sentence_start_positions = p.sentence_start_positions
    sentence_end_positions = p.sentence_end_positions
    sentences_start_positions = p.sentences_start_positions
    sentences_end_positions = p.sentences_end_positions


    try:

        # Process the text with Turbo
        sink = PyCppToPyTurboSink(True)
        retval = turbotextanalysis.analyse_with_tokens('en',
                                                       sentences_words,
                                                       sentence_start_positions,
                                                       sentences_start_positions,
                                                       sentences_end_positions, sink)

        if retval != 0:
            logger.error("ERROR in PyCTurboTextAnalysis analyse")
            logger.error("Return value: {}".format(retval))
            exit()

        tokens_info = sink.get_tokens_info()

        del sink

    except Exception as exp:
        logger.error(
            'Turbo Parser Exception. {}'.format(exp))
        exit()
        return None

    if tokens_info is None or len(tokens_info) == 0:
        return

    output = save_annotation(text, tokens_info, sentences_start_positions, sentences_end_positions, sentence_start_positions, sentence_end_positions)

    end_time = round(time.time() - start_time)
    (t_min, t_sec) = divmod(end_time, 60)
    logger.info('Processing time: {} min {} sec'.format(t_min, t_sec))

    return output

def find_tac_files(dataset_path):
    files_to_process = []

    # Clusters
    for corpus in os.listdir(dataset_path):
        corpus_path = os.path.join(dataset_path, corpus)
        for cluster in os.listdir(corpus_path):
            cluster_path = os.path.join(corpus_path, cluster)
            if os.path.isdir(cluster_path):
                for filename in os.listdir(os.path.join(cluster_path, 'docs')):
                    files_to_process.append(os.path.join(cluster_path, 'docs', filename))
                for filename in os.listdir(os.path.join(cluster_path, 'models')):
                    files_to_process.append(os.path.join(cluster_path, 'models', filename))
    return files_to_process

def find_DUC_files(dataset_path):
    files_to_process = []

    # Clusters
    for corpus in os.listdir(dataset_path):
        if corpus.startswith('DUC'):
            corpus_path = os.path.join(dataset_path, corpus)
            for cluster in os.listdir(corpus_path):
                cluster_path = os.path.join(corpus_path, cluster)
                if os.path.isdir(cluster_path):
                    for filename in os.listdir(os.path.join(cluster_path, 'docs')):
                        files_to_process.append(os.path.join(cluster_path, 'docs', filename))
                    for filename in os.listdir(os.path.join(cluster_path, 'models')):
                        files_to_process.append(os.path.join(cluster_path, 'models', filename))
    return files_to_process

# ------------- Main Code ------------ #

print('Loading TurboParser...')

language = "en"
path = '/opt/TurboParserData/'

turbotextanalysis = None
retval = 0

turbotextanalysis = PyCTurboTextAnalysis()
retval = turbotextanalysis.load_language(language, path)

if retval != 0:
    logger.error('ERROR in PyCTurboTextAnalysis load_language')
    logger.error('Return value: {}'.format(retval))
    exit()


print('Reading folders...')

input_path = '/home/ppb/Data/summarization/datasets/'
files_to_process = find_DUC_files(input_path)

pool = Pool(8)
processes = []

for input_path in files_to_process:
    with codecs.open(input_path, 'r', encoding='utf8') as fp:
        if input_path.endswith('.xml'):
            xml_text = fp.read()
            processes.append(
                (pool.apply_async(process_xml, (xml_text, )), input_path))
            pass
        if input_path.endswith('.txt'):
            text = fp.read()
            # process_txt(text)
            processes.append(
                (pool.apply_async(process_txt, (text, )), input_path))


first_start_time = time.time()
start_time = first_start_time

for i, (p, input_path) in enumerate(processes):
    i += 1
    try:
        text = p.get()

        if input_path.endswith('.xml'):
            # check if the xml_text is well formed
            parser = expat.ParserCreate()
            parser.Parse(text)

        with codecs.open(input_path + '.turboparser', 'w', encoding='utf8') as fp:
            fp.write(text)

    except Exception as e:
        logger.error('File not processed: {} for error: {}'.format(input_path, e))

    if i % 2 == 0:
        t_sec = round(time.time() - start_time)
        (t_min, t_sec) = divmod(t_sec, 60)
        t_sec_comp = round(time.time() - first_start_time) / \
            i * (len(processes) - i)
        (t_min_comp, t_sec_comp) = divmod(t_sec_comp, 60)
        (t_hour_comp, t_min_comp) = divmod(t_min_comp, 60)
        print('Processed {:.0f} of {:.0f} in {:.0f} min and {:.0f} sec. Estimated time to complete: {:.0f} hours {:.0f} min and {:.0f} sec'.format(
            i, len(processes), t_min, t_sec, t_hour_comp, t_min_comp, t_sec_comp), flush=True)
        start_time = time.time()
