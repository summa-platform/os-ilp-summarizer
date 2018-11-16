# This module is part of “Priberam’s Summarizer”, an open-source version of SUMMA’s Summarizer module.
# Copyright 2018 by PRIBERAM INFORMÁTICA, S.A. - www.priberam.com
# You have access to this product in the scope of Project "SUMMA - Project Scalable Understanding of Multilingual Media", Project Number 688139, H2020-ICT-2015.
# Usage subject to The terms & Conditions of the "Priberam  Summarizer OS Software License" available at https://www.priberam.pt/docs/Priberam_Summarizer_OS_Software_License.pdf

import numpy as np

import ad3
#import lpsolve55 as lps

class ExtractiveCoverageSummarizationDecoder():
    '''A decoder for coverage-based extractive summarization.'''

    def __init__(self):
        self.max_words = 100
        self.pruner = False
        self.relax = False

    def decode_cost_augmented(self, parts, scores):
        gold_outputs = np.zeros(len(parts))
        for i, part in enumerate(parts):
            if part.gold:
                gold_outputs[i] = 1
        return self.decode_MIRA(parts, scores, gold_outputs)

    def decode_MIRA(self, parts, scores, gold_outputs):
        cost_type = 'recall'

        index_sentence_parts = [i for i, part in enumerate(parts) if part.type == 'concept']
        p = np.zeros(len(parts))
        if cost_type == 'f1':
            p[index_sentence_parts] = 0.5 - gold_outputs[index_sentence_parts]
            q = 0.5 * np.dot(np.ones(len(index_sentence_parts)),
                             gold_outputs[index_sentence_parts])
        elif cost_type == 'recall':
            p[index_sentence_parts] = -gold_outputs[index_sentence_parts]
            q = np.dot(np.ones(len(index_sentence_parts)),
                       gold_outputs[index_sentence_parts])
            normalize = True
            if normalize:
                p /= q
                q = 1.0
        else:
            raise Exception('Unknown cost type: ' + cost_type)

        scores_cost = scores + p
        predicted_outputs = self.decode(parts, scores_cost)

        cost = np.dot(p, predicted_outputs) + q

        loss = cost + np.dot(scores, predicted_outputs - gold_outputs)

        return predicted_outputs, cost, loss

    def decode(self, parts, scores):

        if self.relax:
            _, predicted_sentences, predicted_concepts = self.summarize_coverage_AD3(parts, scores)
        else:
            _, predicted_sentences, predicted_concepts = self.summarize_coverage(parts, scores)

        predicted_output = np.hstack((predicted_sentences, predicted_concepts))

        return predicted_output

    def summarize_coverage(self, parts, scores):
        '''Obtain the extractive summary of a cluster using a coverage-based
        model'''
        # This is based on Gillick and Favre (2009)

        # Some numbers for the DUC2004 cluster named 'd30042t':
        # 2578 bigram types
        # pruning bigrams with two stopwords reduces to 2279
        # further pruning bigrams which appear in less than 3 documents reduces to 243
        # 167 sentences

        concept_scores = []
        sentence_scores = []
        costs = []
        concept_sentence_list = []
        active_sentences = []
        for i, part in enumerate(parts):
            if part.type == 'sentence':
                sentence_scores.append(scores[i])
                costs.append(part.sentence.num_words())
                active_sentences.append(part.active)
            else:
                concept_scores.append(scores[i])
                concept_sentence_list.append(part.sentence_indices)

        num_concepts = len(concept_scores)
        num_sentences = len(sentence_scores)
        num_variables_total = num_concepts + num_sentences

        # cost_type = 'knapsack'
        cost_type = 'unit'

        # Create LP cluster.
        # num_concepts+num_sentences is the number of problem variables.
        # Convention: variables have concepts first, then sentences.
        lp = lps.lpsolve('make_lp', 0, num_variables_total)

        # Set verbosity level. 3 = only warnings and errors.
        lps.lpsolve('set_verbose', lp, 3)

        # Set objective function.
        # Note the minus sign, since we want to maximize and lp_solve minimizes.
        objective_function_coefficients = [-score for score in concept_scores + sentence_scores]
        lps.lpsolve('set_obj_fn', lp, objective_function_coefficients)

        # Set budget constraint.
        # Knapsack costs are a budget on the number of words.
        # Unit costs are a budget on the number of sentences.
        if cost_type == 'knapsack':
            budget_constraint_coefficients = [0] * num_concepts + costs
        elif cost_type == 'unit':
            budget_constraint_coefficients = [0] * num_concepts + [1] * num_sentences
        else:
            raise NotImplementedError

        lps.lpsolve('add_constraint', lp, budget_constraint_coefficients, lps.LE, 3)

        # Set first set of consistency constraints.
        for i in range(num_concepts):
            for sentence_index in concept_sentence_list[i]:
                # This corresponds to eq. (1) from Gillick and Favre (2009)
                new_constraint_array = [0] * num_variables_total
                # here, +1 is needed because lp_solve starts numbering
                # at 1
                new_constraint_array[i] = -1
                # new_constraint_array[concept_index+1] = -1 # here, +1
                # is needed because lp_solve starts numbering at 1
                new_constraint_array[num_concepts + sentence_index] = 1
                lps.lpsolve('add_constraint', lp, new_constraint_array, lps.LE, 0)

            # This corresponds to eq. (2) from Gillick and Favre (2009)
            new_constraint_array = [0] * num_concepts
            new_constraint_array.extend([1 if index in concept_sentence_list[i] else 0 for index in range(num_sentences)])
            new_constraint_array[i] = -1
            lps.lpsolve('add_constraint', lp, new_constraint_array, lps.GE, 0)

        # Exclude sentences with less than L words.
        for sentence_index in range(num_sentences):
            if not active_sentences[sentence_index]:
                new_constraint_array = [0] * num_variables_total
                new_constraint_array[num_concepts + sentence_index] = 1
                lps.lpsolve('add_constraint', lp, new_constraint_array, lps.EQ, 0)

        lps.lpsolve('set_lowbo', lp, [0] * num_variables_total)
        lps.lpsolve('set_upbo', lp, [1] * num_variables_total)

        if not self.relax:
            lps.lpsolve('set_int', lp, [True] * num_variables_total)
        else:
            lps.lpsolve('set_int', lp, [False] * num_variables_total)

        # lps.lpsolve('write_lp', lp, 'a.lp')

        # Solve the ILP, and call the debugger if something went wrong.
        ret = lps.lpsolve('solve', lp)
        # assert ret == 0

        # Retrieve solution and return
        [solution, _] = lps.lpsolve('get_variables', lp)

        concepts_solution = [0] * num_concepts

        try:
            sentences_solution = solution[num_concepts:]
        except Exception:
            print('Found no solution: {}'.format(solution))
            sentences_solution = [0] * num_concepts
            selected_sentences = []
            return selected_sentences, sentences_solution, concepts_solution

        for concept_index in range(num_concepts):
            val_sum = 0.0
            for sentence_index in concept_sentence_list[concept_index]:
                # If concept score is negative,
                # concept value is the maximum over sentence values.
                # Else it is the sum truncated to one.
                val = sentences_solution[sentence_index]
                val_sum += val
                if val > concepts_solution[concept_index]:
                    concepts_solution[concept_index] = val
            if val_sum > 1.0:
                val_sum = 1.0
            if concept_scores[concept_index] > 0.0:
                concepts_solution[concept_index] = val_sum

        selected_sentences = []
        for i in range(num_sentences):
            value = sentences_solution[i]
            if value > 0.5:
                selected_sentences.append(i)

        lps.lpsolve('delete_lp', lp)

        return selected_sentences, sentences_solution, concepts_solution


    def summarize_coverage_AD3(self, parts, scores):

        concept_scores = []
        sentence_scores = []
        costs = []
        concept_sentence_list = []
        active_sentences = []
        for i, part in enumerate(parts):
            if part.type == 'sentence':
                sentence_scores.append(scores[i])
                costs.append(part.sentence.num_words())
                active_sentences.append(part.active)
            else:
                concept_scores.append(scores[i])
                concept_sentence_list.append(part.sentence_indices)

        num_concepts = len(concept_scores)
        num_sentences = len(sentence_scores)


        # Create a factor graph.
        factor_graph = ad3.PFactorGraph()

        # Create the binary variables.
        binary_variables = []

        # Add concept scores
        for index, score in enumerate(concept_scores):
            variable = factor_graph.create_binary_variable()
            variable.set_log_potential(score)
            binary_variables.append(variable)
            # collect sentence indices
            sentence_indices = concept_sentence_list[index]

        # Add sentence scores and collect sentence variables
        sentence_variables = []
        active_cost_list = []
        index_sentence_variables = {}
        for index, score in enumerate(sentence_scores):
            if active_sentences[index]:
                variable = factor_graph.create_binary_variable()
                variable.set_log_potential(score)
                binary_variables.append(variable)
                index_sentence_variables[index] = len(binary_variables) - 1
                sentence_variables.append(variable)
                active_cost_list.append(costs[index])

        # Constraint #1: At most "maximum_cost" words are selected. Knapsack problem.
        factor_graph.create_factor_knapsack(
            sentence_variables, [False] * len(sentence_variables), np.array(active_cost_list), self.max_words)

        # Set concept-sentence consistency constraints.
        for index in range(num_concepts):
            sentence_indices = concept_sentence_list[index]
            # assert(len(sentence_indices) > 0)

            variables = []
            found_active = False
            for sentence_index in sentence_indices:
                if active_sentences[sentence_index]:
                    found_active = True
                    sentence_variable_index = index_sentence_variables[sentence_index]
                    variables.append(binary_variables[sentence_variable_index])

            if not found_active:
                binary_variables[index].set_log_potential(-1000)
            else:
                variables.append(binary_variables[index])
                negated = [False] * len(variables)
                if len(sentence_indices) == 1:
                    factor_graph.create_factor_logic('XOROUT', variables, negated)
                else:
                    factor_graph.create_factor_logic('OROUT', variables, negated)

        # Now solve the LP with AD3. If relax=False and use_rounding=True,
        # apply a rounding heuristics afterwards.
        factor_graph.set_eta_ad3(0.1)
        factor_graph.adapt_eta_ad3(True)
        factor_graph.set_max_iterations_ad3(1000)

        factor_graph.set_verbosity(1)
        if self.relax:
            value, posteriors, _, _ = factor_graph.solve_lp_map_ad3()
        else:
            value, posteriors, _ = factor_graph.solve_exact_map_ad3()

        sentences_solution = [0] * num_sentences
        for (sentence_index, variable_index) in index_sentence_variables.items():
            sentences_solution[sentence_index] = posteriors[variable_index]

        concepts_solution = [0] * num_concepts
        for concept_index in range(num_concepts):
            val_sum = 0.0
            for sentence_index in concept_sentence_list[concept_index]:
                # If concept score is negative,
                # concept value is the maximum over sentence values.
                # Else it is the sum truncated to one.
                val = sentences_solution[sentence_index]
                val_sum += val
                if val > concepts_solution[concept_index]:
                    concepts_solution[concept_index] = val
            if val_sum > 1.0:
                val_sum = 1.0
            if concept_scores[concept_index] > 0.0:
                concepts_solution[concept_index] = val_sum


        selected_sentences = []
        for i in range(num_sentences):
            value = sentences_solution[i]
            if value > 0.5:
                selected_sentences.append(i)

        return selected_sentences, sentences_solution, concepts_solution
