# CS542 Fall 2021 Homework 3
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
import random
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from worked example
        self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          'dwarf': 4, 'cheered': 5}
        # initial tag weights [shape = (len(tag_dict),)]
        self.initial = np.array([-0.3, -0.7, 0.3])
        # tag-to-tag transition weights [shape = (len(tag_dict),len(tag_dict))]
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        # tag emission weights [shape = (len(word_dict),len(tag_dict))]
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        self.unk_index = -1

    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        tag_vocabulary = set()
        word_vocabulary = set()
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # BEGIN STUDENT CODE
                    document = f.readlines()
                    for line in document:

                        word_and_tags = line.split()
                        for word_and_tag in word_and_tags:

                            last_forward_slash_index = word_and_tag.rfind("/")
                            word_vocabulary.add(word_and_tag[:last_forward_slash_index]) 
                            tag_vocabulary.add(word_and_tag[last_forward_slash_index + 1:]) 

                    # END STUDENT CODE
                    # remove pass keyword when finished
        # create tag_dict and word_dict
        # if you implemented the rest of this
        #  function correctly, these should be formatted
        #  as they are above in __init__
        self.tag_dict = {v: k for k, v in enumerate(tag_vocabulary)}
        self.word_dict = {v: k for k, v in enumerate(word_vocabulary)}

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = [] # doc name + ordinal number of sentence (e.g., ca010)
        sentences = dict()
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # be sure to split documents into sentences here
                    # BEGIN STUDENT CODE
                    sentence_ordinal_number = 0
                    document = f.readlines()
                    for line in document:
                        #get rid of \n, \t, or combinations of them like \n\t or \t\n
                        stripped_line = line.strip()
                        if stripped_line:
                            word_vocabulary = []
                            tag_vocabulary = []

                            #process each word+tag
                            word_and_tags = stripped_line.split()
                            for word_and_tag in word_and_tags:

                                #get the last forward slash of a word+tag, which will guarantee I get the tag
                                #instead of accidentally failing for things like "Doro/thy"
                                last_forward_slash_index = word_and_tag.rfind("/")

                                word_from_word_dict = self.word_dict.get(word_and_tag[:last_forward_slash_index])
                                word_vocabulary.append(word_from_word_dict) 
                                tag_from_tag_dict = self.tag_dict.get(word_and_tag[last_forward_slash_index + 1:])
                                tag_vocabulary.append(tag_from_tag_dict) 

                            #get sentence id and increment count
                            sentence_id = name + str(sentence_ordinal_number)
                            sentence_ids.append(sentence_id)
                            sentence_ordinal_number += 1

                            #get rid of None's
                            word_vocabulary = [self.unk_index if dict_index is None else dict_index for dict_index in word_vocabulary]
                            tag_vocabulary = [self.unk_index if dict_index is None else dict_index for dict_index in tag_vocabulary]

                            #fill dictionaries for a sentence id
                            word_lists[sentence_id] = word_vocabulary
                            tag_lists[sentence_id] = tag_vocabulary
                            sentences[sentence_id] = stripped_line

                    # END STUDENT CODE
                    # remove pass keyword when finished
        return sentence_ids, sentences, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
        best_path = []
        # BEGIN STUDENT CODE

        # initial calculation for first word
        if sentence[0] != self.unk_index:
            v[:,0] = self.initial + self.emission[sentence[0], :]
        else:
            v[:,0] = self.initial + np.zeros(N)
            
        # calculations for rest of sentence
        for t in range(1, T):
            if sentence[t] != self.unk_index:
                results = v[:, t - 1, np.newaxis] + self.transition[:, :] + self.emission[sentence[t], :]
            else:
                results = v[:, t - 1, np.newaxis] + self.transition[:, :] + np.zeros(N)
            v[:, t] = np.max(results, axis=0)
            backpointer[:, t] = np.argmax(results, axis=0)

        # get best path
        best_path_index = np.argmax(v[:, -1])
        best_path = [best_path_index] + best_path
        for t in range(T - 1, 0, -1):
            best_path_index = backpointer[best_path_index, t]
            best_path = [best_path_index] + best_path


        # END STUDENT CODE
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set, dummy_data=None):
        self.make_dicts(train_set)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(train_set)
        if dummy_data is None: # for automated testing: DO NOT CHANGE!!
            Random(0).shuffle(sentence_ids)
            self.initial = np.zeros(len(self.tag_dict))
            self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict)))
            self.emission = np.zeros((len(self.word_dict), len(self.tag_dict)))
        else:
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE

            sentence_as_indexed_word_list = word_lists.get(sentence_id)

            best_predicted_path = self.viterbi(sentence_as_indexed_word_list)
            best_actual_path = tag_lists.get(sentence_id)
            
            #if correct does not equal predicted
            if best_actual_path != best_predicted_path:
                #update initial matrix
                #if first tag is not unknown
                if best_actual_path[0] != -1:
                    self.initial[best_actual_path[0]] += 1
                else:
                    pass
                if best_predicted_path[0] != -1:
                    self.initial[best_predicted_path[0]] -= 1
                else:
                    pass

                #update transition matrix
                for current_tag, next_tag in zip(best_actual_path, best_actual_path[1:]):
                    if current_tag != -1 and next_tag != -1:
                        self.transition[current_tag][next_tag] += 1
                    else:
                        pass
                for current_tag, next_tag in zip(best_predicted_path, best_predicted_path[1:]):
                    if current_tag != -1 and next_tag != -1:
                        self.transition[current_tag][next_tag] -= 1
                    else:
                        pass

                #update emission matrix
                for current_word, current_tag  in zip(sentence_as_indexed_word_list, best_actual_path ):
                    if current_tag != -1 and current_word != -1:
                        self.emission[current_word][current_tag] += 1
                    else:
                        pass
                for current_word, current_tag  in zip(sentence_as_indexed_word_list, best_predicted_path ):
                    if current_tag != -1 and current_word != -1:
                        self.emission[current_word][current_tag] -= 1
                    else:
                        pass

            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'training sentences tagged')

    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set, dummy_data=None):
        results = defaultdict(dict)
        sentence_ids, sentences, tag_lists, word_lists = self.load_data(dev_set)
        if dummy_data is not None: # for automated testing: DO NOT CHANGE!!
            sentence_ids = dummy_data[0]
            sentences = dummy_data[1]
            tag_lists = dummy_data[2]
            word_lists = dummy_data[3]
        for i, sentence_id in enumerate(sentence_ids):
            # BEGIN STUDENT CODE
            sentence_as_indexed_word_list = word_lists.get(sentence_id)

            best_predicted_path = self.viterbi(sentence_as_indexed_word_list)
            best_actual_path = tag_lists.get(sentence_id)

            results[sentence_id]['correct'] = best_actual_path            
            results[sentence_id]['predicted'] = best_predicted_path            


            # END STUDENT CODE
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
        return sentences, results

    '''
    Given results, calculates overall accuracy.
    This evaluate function calculates accuracy ONLY,
    no precision or recall calculations are required.
    '''
    def evaluate(self, sentences, results, dummy_data=False):
        if not dummy_data:
            self.sample_results(sentences, results)
        accuracy = 0.0
        # BEGIN STUDENT CODE
        total_tags = 0
        total_correct_tags = 0

        for correct_or_predicted in results.values():
            correct_path = correct_or_predicted['correct']           
            predicted_path = correct_or_predicted['predicted']
            if self.unk_index in correct_path or self.unk_index in predicted_path:
                pass
            else:
                total_tags += len(correct_path)

                for correct_tag, predicted_tag in zip(correct_path, predicted_path):
                    if correct_tag == predicted_tag:
                        total_correct_tags += 1
        accuracy = total_correct_tags / total_tags
        # END STUDENT CODE
        return accuracy
        
    '''
    Prints out some sample results, with original sentence,
    correct tag sequence, and predicted tag sequence.
    This is just to view some results in an interpretable format.
    You do not need to do anything in this function.
    '''
    def sample_results(self, sentences, results, size=2):
        print('\nSample results')
        results_sample = [random.choice(list(results)) for i in range(size)]
        inv_tag_dict = {v: k for k, v in self.tag_dict.items()}
        for sentence_id in results_sample:
            length = len(results[sentence_id]['correct'])
            correct_tags = [inv_tag_dict[results[sentence_id]['correct'][i]] for i in range(length)]
            predicted_tags = [inv_tag_dict[results[sentence_id]['predicted'][i]] for i in range(length)]
            print(sentence_id,\
                sentences[sentence_id],\
                'Correct:\t',correct_tags,\
                '\n Predicted:\t',predicted_tags,'\n')

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    # pos.train('data_small/train') # train: toy data
    # pos.train('brown_news/train') # train: news data only
    pos.train('brown/train') # train: full data
    # sentences, results = pos.test('data_small/test') # test: toy data
    # sentences, results = pos.test('brown_news/dev') # test: news data only
    sentences, results = pos.test('brown/dev') # test: full data
    print('\nAccuracy:', pos.evaluate(sentences, results))
