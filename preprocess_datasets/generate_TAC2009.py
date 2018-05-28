import os
import shutil
from glob import glob

DATA_PATH = '/home/ppb/data/summarization/orignals/TAC2009/update/TAC2009_Summarization_Documents/UpdateSumm09_test_docs_files'
MODELS_PATH = '/home/ppb/data/summarization/orignals/TAC2009/update/UpdateSumm09_eval/ROUGE/models'
OUTPUT_PATH = '/home/ppb/data/summarization/datasets/TAC2009/'

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
    for track in os.listdir(os.path.join(DATA_PATH, input_cluster)):
        # Only uses the A track from TAC 2008. Track B is the update
        # summarization
        if track.endswith('-A'):
            track_path = os.path.join(DATA_PATH, input_cluster, track)
            for filename in glob(track_path + '/*'):
                shutil.copy(filename, output_documents)
            # rename the files to xml
            for filename in os.listdir(output_documents):
                shutil.move(os.path.join(output_documents, filename),
                            os.path.join(output_documents, filename+'.xml'))

    # copy models
    output_models = os.path.join(output_cluster, 'models')
    os.mkdir(output_models)
    for filename in glob(MODELS_PATH + '/' + input_cluster[:-1] + '*'):
        # again, dont copy track B
        if not '-B' in filename:
            shutil.copy(filename, output_models)
    for filename in os.listdir(output_models):
        shutil.move(os.path.join(output_models, filename),
                    os.path.join(output_models, filename+'.txt'))