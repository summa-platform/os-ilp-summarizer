import os
import pickle
from tabulate import tabulate

from priberam_summarizer.summarization_corpus import SummarizationCorpus
from priberam_summarizer.oracle_summarizer import OracleSummarizer
from priberam_summarizer.evaluation import evaluate
from priberam_summarizer.extractive_summarizer import CoverageExtractiveSummarizer, BasicCoverageExtractiveSummarizer

NUM_JOBS = 1
PICKLE_DATASET = False

if not os.path.exists('DUC2003.pkl'):
    trainset = SummarizationCorpus(n_jobs=NUM_JOBS)
    trainset.read_corpus('/home/ppb/Data/summarization/datasets/DUC2003/', max_folders=10)
    trainset.compute_tfidf()
    summarizer = OracleSummarizer(trainset.vector_space, n_jobs=NUM_JOBS)
    train_oracles = summarizer.create_oracles(trainset.clusters)
    if PICKLE_DATASET:
        pickle.dump((trainset, train_oracles), open('DUC2003.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
else:
    trainset, train_oracles = pickle.load(open('DUC2003.pkl', 'rb'))


results = []

metrics = evaluate(trainset.clusters, train_oracles)

results.append([
    'Dailymail Train (Oracle)',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])


summarizer = BasicCoverageExtractiveSummarizer(trainset.vector_space, n_jobs=NUM_JOBS)
summaries = summarizer.test(trainset.clusters)
metrics = evaluate(trainset.clusters, summaries)

results.append([
    'DailyMail Train (Basic. Coverage)',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))

if not os.path.exists('DUC2004.pkl'):
    devset = SummarizationCorpus(n_jobs=NUM_JOBS)
    devset.read_corpus('/home/ppb/Data/summarization/datasets/DUC2004/', max_folders=10)
    devset.compute_tfidf()
    summarizer = OracleSummarizer(trainset.vector_space, n_jobs=NUM_JOBS)
    dev_oracles = summarizer.create_oracles(devset.clusters)
    pickle.dump((devset, dev_oracles), open('DUC2004.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
else:
    devset, dev_oracles = pickle.load(open('DUC2004.pkl', 'rb'))

summarizer = CoverageExtractiveSummarizer(trainset.vector_space, n_jobs=NUM_JOBS)
model = summarizer.train(trainset.clusters, train_oracles, num_epochs=10)
model.save('summarizer_10epochs.model')

summaries = summarizer.test(trainset.clusters)
metrics = evaluate(trainset.clusters, summaries)

results.append([
    'Dailymail Train (Extractive) 10 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

summaries = summarizer.test(devset.clusters)
metrics = evaluate(devset.clusters, summaries)

results.append([
    'Dailymail Dev (Extractive) 10 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))

print('Train.')
model15 = summarizer.train(trainset.clusters, train_oracles, num_epochs=5)
print('save.')
model15.save('summarizer_15epochs.model')

print('test.')
summaries = summarizer.test(trainset.clusters)
print('Evaluate.')
metrics = evaluate(trainset.clusters, summaries)

results.append([
    'Dailymail Train (Extractive) 15 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print('TestDev.')
summaries = summarizer.test(devset.clusters)
metrics = evaluate(devset.clusters, summaries)

results.append([
    'Dailymail Dev (Extractive) 15 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))

model = summarizer.train(trainset.clusters, train_oracles, num_epochs=5)
model.save('summarizer_20epochs.model')

summaries = summarizer.test(trainset.clusters)
metrics = evaluate(trainset.clusters, summaries)

results.append([
    'Dailymail Train (Extractive) 20 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

summaries = summarizer.test(devset.clusters)
metrics = evaluate(devset.clusters, summaries)

results.append([
    'Dailymail Dev (Extractive) 20 epochs',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))

if not os.path.exists('DUC2003.pkl'):
    corpus = SummarizationCorpus(n_jobs=NUM_JOBS)
    corpus.read_corpus('/home/ppb/Data/summarization/datasets/DUC2003/', max_folders=10)
    print('read_corpus')
    corpus.compute_tfidf()
    print('compute_tf_idf')
    summarizer = OracleSummarizer(corpus.vector_space, n_jobs=NUM_JOBS)
    oracles = summarizer.create_oracles(corpus.clusters)
    print('oracle_done')
    pickle.dump((corpus, oracles), open('DUC2003.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    print('pickle done')
else:
    corpus, oracles = pickle.load(open('DUC2003.pkl', 'rb'))

print('saving oracle')
save_path = 'output/oracle_extractive'
for summary in oracles:
    with open(os.path.join(save_path, summary.name + '.summary'), 'w', encoding='utf8') as fp:
        fp.write(summary.get_text() + '\n')

print('evaluating')
metrics = evaluate(corpus.clusters, oracles)

results.append([
    'Dailymail Test (Oracle)',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

#model.load('summarizer_15epochs.model')

summarizer = CoverageExtractiveSummarizer(corpus.vector_space, model=model15, n_jobs=NUM_JOBS)
summaries = summarizer.test(corpus.clusters)
metrics = evaluate(corpus.clusters, summaries)

save_path = 'output/cov_extractive'
for summary in summaries:
    with open(os.path.join(save_path, summary.name + '.summary'), 'w', encoding='utf8') as fp:
        fp.write(summary.get_text() + '\n')

results.append([
    'Dailymail Test (Extractive)',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])


summarizer = BasicCoverageExtractiveSummarizer(corpus.vector_space, n_jobs=NUM_JOBS)
summaries = summarizer.test(corpus.clusters)

save_path = 'output/basic_cov_extractive'
for summary in summaries:
    with open(os.path.join(save_path, summary.name + '.summary'), 'w', encoding='utf8') as fp:
        fp.write(summary.get_text() + '\n')

metrics = evaluate(corpus.clusters, summaries)

results.append([
    'Dailymail Test (Basic. Coverage)',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])


print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))


'''

metrics = evaluate_against_oracle(oracles, oracles)

results.append([
    'TAC 2009 Oracle aginst Oracle',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])


metrics = evaluate_against_oracle(oracles, summaries)

results.append([
    'TAC 2009 Extractive aginst Oracle',
    metrics['rouge_1_precision'],
    metrics['rouge_1_recall'],
    metrics['rouge_1_f_score'],
    '',
    metrics['rouge_2_precision'],
    metrics['rouge_2_recall'],
    metrics['rouge_2_f_score'],
    '',
    metrics['rouge_l_precision'],
    metrics['rouge_l_recall'],
    metrics['rouge_l_f_score'],
    '',
    metrics['rouge_su4_precision'],
    metrics['rouge_su4_recall'],
    metrics['rouge_su4_f_score']])

print(tabulate(results, headers=[
    'Dataset/System',
    'R-1 P', 'R-1 R', 'R-1 F', '',
    'R-2 P', 'R-2 R', 'R-2 F', '',
    'R-L P', 'R-L R', 'R-L F', '',
    'R-SU4 P', 'R-SU4 R', 'R-SU4 F'
]))
'''
