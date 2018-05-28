import os
import shutil
import xml
from xml.sax.saxutils import escape
from xml.parsers.expat import ParserCreate, ExpatError, errors

datasets = list()

data_path = '/home/ppb/data/summarization/orignals/CNN/training'
output_path = '/home/ppb/data/summarization/datasets/CNN_train'
datasets.append((data_path, output_path))

# Erase any data remaining from past runs
if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)

for doc_filename in os.listdir(os.path.join(data_path, 'docs')):
    if doc_filename.endswith('.doc.txt'):

        text = []
        doc_path = os.path.join(data_path, 'docs', doc_filename)
        with open(doc_path, 'r', encoding='latin1') as f:
            headline = f.readline().strip()[2:]
            for line in f:
                line = line.strip()
                if len(line) >= 3:
                    # remove empty lines and the number (1,2,3) starting each line
                    text.append(line[2:])

        model_path = os.path.join(data_path, 'hlights', doc_filename[:-8] + '.hlights.txt')

        if not os.path.exists(model_path):
            print('Summary not found: {}'.format(model_path))
            continue

        with open(model_path, 'r', encoding='latin1') as f:
            summary = f.read().strip()

        if len(text) == 0:
            print('Document empty: {}'.format(doc_path))
            continue

        if len(summary) == 0:
            print('Summary empty: {}'.format(model_path))
            continue

        output_cluster = os.path.join(output_path, doc_filename[:-8])
        os.mkdir(output_cluster)

        output_documents = os.path.join(output_cluster, 'docs')
        os.mkdir(output_documents)


        output_filename = os.path.join(output_documents, doc_filename[:-8] + '.xml')

        xml = '<DOC id="{}" >\n'.format(escape(doc_filename[:-8]))
        xml += '<HEADLINE>\n'
        xml += headline+'\n'
        xml += '</HEADLINE>\n<TEXT>\n'
        xml += escape('\n'.join(text))
        xml += '\n</TEXT>\n</DOC>'

        # check if the xml is good
        try:
            parser = ParserCreate()
            parser.Parse(xml)
        except ExpatError as err:
            print('Problem generating xml file {}'.format(output_filename))
            print("Error:", errors.messages[err.code])

        with open(output_filename, 'w', encoding='utf8') as f:
            # should call the proper xml contructor, but this solved for the corpus
            f.write(xml)

        output_models = os.path.join(output_cluster, 'models')
        os.mkdir(output_models)

        output_summary = os.path.join(output_models, doc_filename[:-8] + '.hlights.txt')
        with open(output_summary, 'w', encoding='utf8') as f:
            f.write(summary)

#=====================================================#

data_path = '/home/ppb/data/summarization/orignals/CNN/test'
output_path = '/home/ppb/data/summarization/datasets/CNN_test'

# Erase any data remaining from past runs
if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)

for doc_filename in os.listdir(os.path.join(data_path, 'docs')):
    if doc_filename.endswith('.doc.txt'):

        text = []
        doc_path = os.path.join(data_path, 'docs', doc_filename)
        with open(doc_path, 'r', encoding='latin1') as f:
            headline = f.readline().strip()[2:]
            for line in f:
                line = line.strip()
                if len(line) >= 3:
                    # remove empty lines and the number (1,2,3) starting each line
                    text.append(line)

        model_path = os.path.join(data_path, 'hlights', doc_filename[:-8] + '.hlights.txt')

        if not os.path.exists(model_path):
            print('Summary not found: {}'.format(model_path))
            continue

        with open(model_path, 'r', encoding='latin1') as f:
            summary = f.read().strip()

        if len(text) == 0:
            print('Document empty: {}'.format(doc_path))
            continue

        if len(summary) == 0:
            print('Summary empty: {}'.format(model_path))
            continue

        output_cluster = os.path.join(output_path, doc_filename[:-8])
        os.mkdir(output_cluster)

        output_documents = os.path.join(output_cluster, 'docs')
        os.mkdir(output_documents)


        output_filename = os.path.join(output_documents, doc_filename[:-8] + '.xml')

        xml = '<DOC id="{}" >\n'.format(escape(doc_filename[:-8]))
        xml += '<HEADLINE>\n'
        xml += headline+'\n'
        xml += '</HEADLINE>\n<TEXT>\n'
        xml += escape('\n'.join(text))
        xml += '\n</TEXT>\n</DOC>'

        # check if the xml is good
        try:
            parser = ParserCreate()
            parser.Parse(xml)
        except ExpatError as err:
            print('Problem generating xml file {}'.format(output_filename))
            print("Error:", errors.messages[err.code])

        with open(output_filename, 'w', encoding='utf8') as f:
            # should call the proper xml contructor, but this solved for the corpus
            f.write(xml)

        output_models = os.path.join(output_cluster, 'models')
        os.mkdir(output_models)

        output_summary = os.path.join(output_models, doc_filename[:-8] + '.hlights.txt')
        with open(output_summary, 'w', encoding='utf8') as f:
            f.write(summary)

