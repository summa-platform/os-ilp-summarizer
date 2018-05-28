import os
import shutil
import nltk
from fuzzywuzzy import fuzz
from xml.sax.saxutils import escape
from xml.parsers.expat import ParserCreate, ExpatError, errors
import itertools

data_path = '/home/ppb/Data/summarization/orignals/CNN_Dailymail/cnn/test'
output_path = '/home/ppb/Data/summarization/datasets/CNN_Dailymail/cnn/test'

# Erase any data remaining from past runs
if os.path.exists(output_path):
    shutil.rmtree(output_path)

os.mkdir(output_path)

output_documents = os.path.join(output_path, 'docs')
os.mkdir(output_documents)

output_models = os.path.join(output_path, 'models')
os.mkdir(output_models)


f_source = open(os.path.join(output_path, 'sentences.txt'), 'w', encoding='utf8')
f_target = open(os.path.join(output_path, 'highlights.txt'), 'w', encoding='utf8')

for doc_filename in os.listdir(data_path):

    text = {}
    field = 'UNK'
    try:
        with open(os.path.join(data_path, doc_filename), 'r', encoding='utf8') as fp:
            for line in fp:
                line = line.strip()
                if line.startswith('[SN]'):
                    field = line[4:-4]
                else:
                    text[field] = text.get(field, []) + [line]

            if 'HighlightsOrg' not in text or 'StoryOrg' not in text:
                print('Not find highlights or story for {}'.format(doc_filename))

            text_sentences = nltk.sent_tokenize('\n'.join(text['StoryOrg']))
            highlights = nltk.sent_tokenize('\n'.join(text['HighlightsOrg']))

            for highlight in highlights:
                best_match = [0, '']
                for sentence in text_sentences:
                    score = fuzz.partial_ratio(sentence.lower(), highlight.lower())
                    if score >= best_match[0]:
                        best_match = [score, sentence]

                for sentenceA, sentenceB in itertools.permutations(text_sentences, 2):
                    sentence = sentenceA + ' ' + sentenceB
                    score = fuzz.partial_ratio(sentence.lower(), highlight.lower())
                    if score >= best_match[0]:
                        best_match = [score, sentence]

                for sentenceA, sentenceB, sentenceC in itertools.permutations(text_sentences, 3):
                    sentence = sentenceA + ' ' + sentenceB + ' ' + sentenceC
                    score = fuzz.partial_ratio(sentence.lower(), highlight.lower())
                    if score >= best_match[0]:
                        best_match = [score, sentence]

                print(doc_filename)
                # At least 50% match
                if best_match[0] >= 50:
                    f_source.write(best_match[1].replace('\n', ' ') + '\n')
                    f_target.write(highlight.replace('\n', ' ') + '\n')


            output_filename = os.path.join(output_documents, doc_filename + '.xml')

            xml = '<DOC id="{}" >\n'.format(escape(doc_filename))
            xml += '<HEADLINE>\n'
            xml += escape(text.get('TitleTokenized', [''])[0]) +'\n'
            xml += '</HEADLINE>\n<TEXT>\n'
            xml += escape('\n'.join(text['StoryOrg']))
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

            output_summary = os.path.join(output_models, doc_filename + '.hlights.txt')
            with open(output_summary, 'w', encoding='utf8') as f:
                f.write('\n'.join(text['HighlightsOrg']))
    except Exception as exp:
        print('Exception {} at {}'.format(exp, doc_filename))