import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from math import sqrt

from .extractive_summarization_decoder import ExtractiveCoverageSummarizationDecoder
from .parts import SentencePart, ConceptPart

from .document import Document

class OracleSummarizer():

    def __init__(self, vector_space, max_num_words=100, min_sentence_length=3, max_sentence_length=100, use_stemmer=True, lower_case=True, n_jobs=-1):
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.vector_space = vector_space
        self.max_num_words = max_num_words
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.use_stemmer = use_stemmer
        self.lower_case = lower_case

    def create_oracles(self, clusters):
        oracles = []
        if self.n_jobs == 1:
            for cluster in clusters:
                oracles.append(self.create_oracle_summary(cluster, cluster.reference_summaries))
        else:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = []
                for cluster in clusters:
                    futures.append(executor.submit(self.create_oracle_summary, cluster, cluster.reference_summaries))

                for future in futures:
                    oracles.append(future.result())
        return oracles


    def create_oracle_summary(self, cluster, reference_summaries):
        '''Create summary oracle from existing reference summaries. The
        oracles are the ones that (approximately) maximize rouge and are
        extractive summaries.'''
        parts = []

        # compute unigram and bigram count for the summary
        bigrams = Counter()
        bigram_idfs = self.vector_space.bigram_idfs

        for document in reference_summaries:
            for sentence in document.sentences:
                tokens = [token.lemma if self.use_stemmer else token.word for token in sentence.tokens]
                tokens = [token.lower() if self.lower_case else token for token in tokens]
                tokens = ['__start__'] + tokens + ['__end__']
                sentence_bigrams = Counter(zip(tokens, tokens[1:]))

                # update the count in document
                bigrams.update(sentence_bigrams)

        # compute tf_idf for bigrams
        bigram_tfidf = {key: freq * bigram_idfs.get(key, bigram_idfs['__oov__']) for key, freq in bigrams.items()}
        norm_factor = sqrt(sum([value * value for value in bigram_tfidf.values()]))
        bigram_tfidf = {key: value / norm_factor if norm_factor > 0 else value for key, value in bigram_tfidf.items()}

        sentence_scores = []
        concept_sentences = {}
        for sentence_index, sentence in enumerate(cluster.sentences):
            part = SentencePart(sentence)
            sentence_score = 0
            for concept in set(bigram_tfidf).intersection(set(sentence.bigram_tfidf)):
                concept_sentences[concept] = concept_sentences.get(concept, []) + [sentence_index]
                sentence_score += bigram_tfidf[concept]
            sentence_scores.append(sentence_score)
            part.active = True if sentence.num_words() >= self.min_sentence_length and sentence.num_words() <= self.max_sentence_length else False
            part.value = sentence_score
            parts.append(part)

        # Concept scores are based on summary tf-idf concepts
        for concept in sorted(bigram_tfidf):
            part = ConceptPart(concept, bigram_tfidf[concept], concept_sentences.get(concept, []))
            parts.append(part)

        # normalize sentence scores
        if sum(sentence_scores) > 0:
            for part in parts:
                if part.type == 'sentence':
                    part.value = part.value / sum(sentence_scores)

            scores = [part.value for part in parts]
            decoder_aux = ExtractiveCoverageSummarizationDecoder()
            selected_sentences, _, predicted_concepts = decoder_aux.summarize_coverage(parts, scores)
        else:
            # pick the first sentence in case of no vocabulary is shared
            selected_sentences = [0]

        # Final summary and its score.
        summary = Document()
        #summary.name = 'Oracle: ' + cluster.name
        summary.name = cluster.name
        for selected_sentence in selected_sentences:
            summary.body.append(cluster.sentences[selected_sentence])

        setattr(summary, 'selected_sentences', selected_sentences)
        #setattr(summary, 'predicted_concepts', predicted_concepts)

        #print('Finished computing oracle for cluster {} ...'.format(cluster.name))
        return summary