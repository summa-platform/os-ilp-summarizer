import os
import shutil
from glob import glob

DATA_PATH = '/home/ppb/Data/summarization/originals/DUC2004/duc04.results.data/past_duc/duc2004/testdata/tasks1and2/duc2004.tasks1and2.docs/t1.2/docs'
MODELS_PATH = '/home/ppb/Data/summarization/originals/DUC2004/duc04.results.data/past_duc/duc2004/results/ROUGE/eval/models/2'
OUTPUT_PATH = '/home/ppb/Data/summarization/datasets/DUC2004/'

# Erase any data remaining from past runs
if os.path.exists(OUTPUT_PATH):
    shutil.rmtree(OUTPUT_PATH)

os.mkdir(OUTPUT_PATH)

for input_cluster in os.listdir(DATA_PATH):
    output_cluster = os.path.join(OUTPUT_PATH, input_cluster)
    os.mkdir(output_cluster)
    output_documents = os.path.join(output_cluster, 'docs')
    os.mkdir(output_documents)

    # copy documents
    for filename in os.listdir(os.path.join(DATA_PATH, input_cluster)):
        shutil.copy(os.path.join(DATA_PATH, input_cluster, filename), output_documents)
        # rename the files to xml
    for filename in os.listdir(output_documents):
        shutil.move(os.path.join(output_documents, filename),
                    os.path.join(output_documents, filename+'.xml'))

    # copy models
    output_models = os.path.join(output_cluster, 'models')
    os.mkdir(output_models)
    for filename in glob(os.path.join(MODELS_PATH, input_cluster[:-1].upper() + '.M.100.*')):
        shutil.copy(filename, output_models)
    for filename in os.listdir(output_models):
        shutil.move(os.path.join(output_models, filename),
                    os.path.join(output_models, filename+'.txt'))