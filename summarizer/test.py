import os

from priberam_summarizer.summarization_corpus import SummarizationCorpus
from priberam_summarizer.extractive_summarizer import CoverageExtractiveSummarizer, BasicCoverageExtractiveSummarizer
from priberam_summarizer.linear_model import LinearModel
from priberam_summarizer.vector_space_model import VectorSpaceModel
import nltk

NUM_JOBS = 1


# Model
model = LinearModel()
model.load('summarizer.model')

# VectorSpace
vector_space = VectorSpaceModel()
vector_space.load_vector_space_model('vector_space.pickle')

# Summarizer
summarizer = CoverageExtractiveSummarizer(vector_space, model=model, n_jobs=1)

trainset = SummarizationCorpus(n_jobs=NUM_JOBS)
trainset.read_corpus('/home/ppb/Data/summarization/datasets/TAC2008/', max_folders=2)
trainset.compute_tfidf()
#vector_space = trainset.vector_space
#vector_space.save_vector_space_model('vector_space.pickle')
summaries = summarizer.test(trainset.clusters)
for summary in summaries:    
        print(summary.get_text() + '\n')

