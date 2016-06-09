#!/usr/bin/env python2

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import ast

from wordcloud import WordCloud
from pprint import pprint

sys.path.append("..") # for utils
from utils import grey_color_func

def post_process():

    #with open('clda_data/out_prism', 'r') as fin:
    #    phi_prism = [np.array(ast.literal_eval(line.strip())) for line in fin]
    #phi_prism = np.array(phi_prism)

    #theta_pb = np.load('/tmp/peircebayes/avg_samples.npz')
    #theta_pb = np.load('/home/rares/Desktop/peircebayes_all_no_sampling/last_sample.npz')
    theta_pb = np.load('data/avg_samples.npz')
    phi = theta_pb['arr_1']
    print phi.shape

    vocab = pickle.load(open('data/vocab.pkl', 'r'))
    inv = dict((v, k) for k, v in vocab.iteritems())

    axis = 1
    index = list(np.ix_(*[np.arange(i) for i in phi.shape]))
    index[axis] = phi.argsort(axis)
    a = phi[index][:,-20:]
    counts = np.rint(a/np.sum(a, axis=1).reshape(-1,1)*1000).tolist()
    idx_l = index[axis][:,-20:].tolist()
    words = [[inv[i] for i in subl] for subl in idx_l]
    #pprint(words)

    index_prism = list(np.ix_(*[np.arange(i) for i in phi_prism.shape]))
    index_prism[axis] = phi_prism.argsort(axis)
    a_prism = phi_prism[index_prism][:,-20:]
    idx_l_prism = index_prism[axis][:,-20:].tolist()
    words_prism = [[inv[i] for i in subl] for subl in idx_l_prism]

    #pprint(words_prism)

    # topic 1
    freq1 = list(reversed(zip(words[0], list(a[0,:]))))
    # topic 2
    freq2 = list(reversed(zip(words[1], list(a[1,:]))))

    # topic 1
    #freq1_prism = list(reversed(zip(words_prism[19], list(a_prism[19,:]))))
    # topic 2
    #freq2_prism = list(reversed(zip(words_prism[18], list(a_prism[18,:]))))


    wc = WordCloud(background_color="white", width=400, height=400,
        random_state=1234).fit_words(freq1)

    plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
    plt.axis("off")
    plt.savefig('data/topic_1.pdf', format='pdf')
    plt.close()

    plt.imshow(wc.fit_words(freq2).recolor(color_func=grey_color_func, random_state=3))
    plt.axis("off")
    plt.savefig('data/topic_2.pdf', format='pdf')
    plt.close()

    #plt.imshow(wc.fit_words(freq1_prism).recolor(color_func=grey_color_func, random_state=3))
    #plt.axis("off")
    #plt.savefig('data/prism_topic_1.pdf', format='pdf')
    #plt.close()

    #plt.imshow(wc.fit_words(freq2_prism).recolor(color_func=grey_color_func, random_state=3))
    #plt.axis("off")
    #plt.savefig('data/prism_topic_2.pdf', format='pdf')
    #plt.close()


def main():
    post_process()

if __name__=='__main__':
    main()
