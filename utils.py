import os
import shutil
import subprocess
import shlex
import random
import itertools

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import multinomial
from numpy.random import dirichlet

def grey_color_func(word, font_size, position, orientation,
    random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(1, 60)

def plot_topics(T,phi,fname):
    f, axs = plt.subplots(1,T+1,figsize=(15,1))
    ax = axs[0]
    ax.text(0,0.4, "Topics: ", fontsize = 16)
    ax.axis("off")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    for (ax, (i,phi_t)) in zip(axs[1:], enumerate(phi)):
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.imshow(phi_t, cmap='Greys_r', interpolation='none')
    f.savefig('{}.pdf'.format(fname), format='pdf')
    plt.close()

def call_cmd(cmd_str, out=None, err=None):
    devnull = open('/dev/null', 'w')
    if out is None:
        out = devnull
    if err is None:
        err = devnull
    p1 = subprocess.Popen(shlex.split(cmd_str),
        stdout=out, stderr=err)
    p1.wait()

def rewrite_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        shutil.rmtree(dir_path)
        os.makedirs(dir_path)

def generate_lda(T, W, D, N, phi, alpha):
    # sample theta from alpha
    theta = [dirichlet(alpha*np.ones(T)) for i in range(D)]
    B = []
    # sample documents from theta and phi
    for d in range(D):
        doc = np.zeros(W)
        theta_sample = multinomial(N, theta[d])
        for t,count in enumerate(theta_sample):
            doc += multinomial(count, phi[int(t)].flatten())
        B.append(doc)
    return B

def write_pb(data_dir, idx, W, T, D, alpha, beta, B, write_params=True):
    fname = os.path.join(data_dir, 'lda_{}.pb'.format(idx))
    with open(fname, 'w') as fout:
        if write_params:
            fout.write('''% LDA artificial data

% needed to ground constraints
:- enforce_labeling(true).

pb_dirichlet({}, mu, {}, {}).
pb_dirichlet({}, phi, {}, {}).

generate(Doc, Token) :-
    Topic in 1..{},
    mu(Topic,Doc),
    phi(Token,Topic).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [generate(Doc, Token)]
).

'''.format(alpha, T, D, beta, W, T, T))
        else:
            fout.write('''% LDA artificial data

% needed to ground constraints
:- enforce_labeling(true).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [generate(Doc, Token)]
).

''')
        for i,doc in enumerate(B):
            word_list = ['(w({}), {})'.format(j+1,int(count))
                for j,count in enumerate(doc)
                if count>0]
            str_word_list = '['+','.join(word_list)+']'
            fout.write('observe(d({}), {}).\n'.format(i+1, str_word_list))

def write_txt(data_dir, idx, B, T, alpha, beta, write_params=True):
    fname2 = os.path.join(data_dir, 'lda_{}.txt'.format(idx))
    with open(fname2, 'w') as fout:
        for i,doc in enumerate(B):
            for j,count in enumerate(doc):
                if count>0:
                    fout.write('{} {} {}\n'.format(i+1, j+1, count))
    with open(os.path.join(data_dir, 'gibbs_lda.R'), 'w') as fout:
        fout.write('''
# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)
''')
        if write_params:
            fout.write('''
K = {}
alpha = {}
beta = {}
# iter = ...
# seed = 
'''.format(T, alpha, beta))
        fout.write('''set.seed(seed)
run_gibbs <- function(f){{
  print(f)
  data = read.table(f, quote="\\\"")
  lapply(sample(1:5000, 1, replace=F), function(s){{
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="Gibbs", control=list(
      seed = s,
      alpha = alpha,
      delta = beta, # this is beta
      keep = 1, # save log likelihood every iteration
      iter = iter,
      burnin = 0
      ))
    model
    }})
}}        
''')    
    with open(os.path.join(data_dir, 'vem_lda.R'), 'w') as fout:
        fout.write('''
# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)
''')
        if write_params:
            fout.write('''
K = {}
alpha = {}
beta = {}
# iter = ...
# seed = 
'''.format(T, alpha, beta))
        fout.write('''set.seed(seed)
run_vem <- function(f){{
  print(f)
  data = read.table(f, quote="\\\"")
  lapply(sample(1:5000, 1, replace=F), function(s){{
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="VEM", control=list(
        seed = seed,
        alpha = alpha,
        em = list(iter.max=iter),
        keep = 1 # save log likelihood every iteration
  ))
  model
  }})
}}        
''')                       

def write_church(data_dir, idx, B, alpha,beta,D,T,W,N, n_samples, lag):
    words = []
    for b in B:
        doc = []
        for j,el in enumerate(b.flatten()):
            doc += [j]*el
        words.append(doc)
    fname2 = os.path.join(data_dir, 'lda_church1_{}.md'.format(idx))
    with open(fname2, 'w') as fout:
        fout.write('''; LDA artificial example
;;;fold: factor-equal?
(define (factor-eq x y)
  (factor (if (equal? x y) 0.0 -1000))
  #t)

(define (factor-equal? xs ys)
  (if (and (null? xs) (null? ys))
      #t
      (and (factor-eq (first xs) (first ys))
           (factor-equal? (rest xs) (rest ys)))))
;;;
''')
        voc_str = ' '.join(['w{}'.format(i) for i in range(1,W+1)])
        t_str = ' '.join(['{}'.format(i) for i in range(T)])
        fout.write('''
(define vocabulary '({}))

(define topics '({}))

(define doc-length {})

'''.format(voc_str, t_str, N))
        # write data
        for doc_id, doc in enumerate(words):
            doc_str = ' '.join(['w{}'.format(i+1) for i in doc])
            fout.write('''(define doc{} '({}))\n'''.format(doc_id+1, doc_str))
        # aux for ll
        doc_str = ' '.join(['doc{}'.format(i) for i in range(1,N+1)])
        fout.write('''
(define docs (list {}))

(define doc->wid
  (lambda (word) (list-index vocabulary word)))

(define docs->wid
  (lambda (doc) (map doc->wid doc)))

(define w (map docs->wid docs))
'''.format(doc_str))

        # write query
        constraint_str = '\n'.join([
            "(factor-equal? (document->words 'doc{}) doc{})".format(i,i)
            for i in range(1,D+1)])
        fout.write('''; query
(define samples
  (mh-query
   {} {}

   (define document->mixture-params
     (mem (lambda (doc-id) (dirichlet (make-list (length topics) {})))))

   (define topic->mixture-params
     (mem (lambda (topic) (dirichlet (make-list (length vocabulary) {})))))

   (define document->topics
     (mem (lambda (doc-id)
            (repeat doc-length
                    (lambda () (multinomial topics (document->mixture-params doc-id)))))))

   (define document->words
     (mem (lambda (doc-id)
            (map (lambda (topic)
                   (multinomial vocabulary (topic->mixture-params topic)))
                 (document->topics doc-id)))))

   (define z
     (map document->topics '({})))

   ; ll per document
   (define (doc-ll doc-w doc-z doc-idx)
     (if (and (null? doc-w) (null? doc-z))
       0
       (+ (log (list-ref (topic->mixture-params (first doc-z)) (first doc-w))) ; phi
          ; no theta ;(log (list-ref (document->mixture-params doc-idx) (first doc-z)))    ; theta
          (doc-ll (rest doc-w) (rest doc-z) doc-idx))                          ; recurse
     )
   )

   ; log-likelihood function
   (define (lda-ll w z doc-idx)
     (if (and (null? w) (null? z))
       0
       (+ (doc-ll (first w) (first z))
          (lda-ll (rest w) (rest z) (+ doc-idx 1)))
     )
   )

   ; sample the ll
   (lda-ll w z 0)

   (and
    {}
    )))

(write-csv (list samples) "ll_church1.csv" " ")
'''.format(n_samples,lag,alpha,beta,doc_str,constraint_str))

def write_church2(data_dir, idx, B, alpha,beta,D,T,W,n_samples,lag):
    words = []
    for b in B:
        doc = []
        for j,el in enumerate(b.flatten()):
            doc += [j]*el
        words.append(doc)
    fname2 = os.path.join(data_dir, 'lda_church2_{}.md'.format(idx))
    with open(fname2, 'w') as fout:
        fout.write('''
(define (index-in x xs)
  (define (loop x k rst)
    (if (is_null rst) k
      (if (equal? (first rst) x) k
        (loop x (+ k 1) (rest rst)))))
    (loop x 0 xs))

(define (word-factor i distr)
  (let ((p (list-ref distr i)))
    (factor (log p))))
''')
        t_str = ' '.join(['{}'.format(i) for i in range(T)])
        voc_str = ' '.join(['w{}'.format(i) for i in range(1,W+1)])
        fout.write('''
(define number-of-topics {})
(define topics '({}))
(define vocabulary '({}))
'''.format(T,t_str,voc_str))
        doc_str = ''
        for doc_id, doc in enumerate(words):
            doc_str += '('+' '.join(['w{}'.format(i+1) for i in doc])+')\n'
        fout.write('''
(define documents
  '(
  {}
  ))
'''.format(doc_str))
        alpha_str = np.array_str(alpha*np.ones(T))[1:-1]
        beta_str = np.array_str(beta*np.ones(W))[1:-1]
        fout.write('''
(define samples
  (mh-query
   {} {}

   (define topic-word-distributions
     (repeat number-of-topics
             (lambda () (dirichlet '({})))))

   (define process
     (sum (map
      (lambda (document)
        (let* ((topic-selection-distr (dirichlet '( 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.))))
          (sum (map (lambda (word)
                 (let* ((sampled-topic (multinomial topics topic-selection-distr))
                        (idx (index-in word vocabulary)))
                   ; no theta
                   ;(+ (log (list-ref (list-ref topic-word-distributions sampled-topic) idx))
                   ;   (log (list-ref topic-selection-distr sampled-topic)))
                   (log (list-ref (list-ref topic-word-distributions sampled-topic) idx))
                   ))
          document))))
      documents)))

   process

   #t))

(write-csv (list samples) "ll_church2.csv" " ")
'''.format(n_samples, lag,beta_str, alpha_str))

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def vertical_topic(width, topic_index, document_length):
    """
    Generate a topic whose words form a vertical bar.
    """
    m = np.zeros((width, width))
    m[:, topic_index] = int(document_length / width)
    return m.flatten()

def horizontal_topic(width, topic_index, document_length):
    """
    Generate a topic whose words form a horizontal bar.
    """
    m = np.zeros((width, width))
    m[topic_index, :] = int(document_length / width)
    return m.flatten()

def gen_word_distribution(n_topics=10, document_length=100):
    """
    Generate a word distribution for each of the n_topics.
    """
    width = n_topics / 2
    vocab_size = width ** 2
    m = np.zeros((n_topics, vocab_size))

    for k in range(width):
        m[k,:] = vertical_topic(width, k, document_length)

    for k in range(width):
        m[k+width,:] = horizontal_topic(width, k, document_length)

    m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

    return m

def gen_document(word_dist, n_topics, vocab_size, length, alpha):
    """
    Generate a document:
        1) Sample topic proportions from the Dirichlet distribution.
        2) Sample a topic index from the Multinomial with the topic
           proportions from 1).
        3) Sample a word from the Multinomial corresponding to the topic
           index from 2).
        4) Go to 2) if need another word.
    """
    theta = np.random.mtrand.dirichlet([alpha] * n_topics)
    v = np.zeros(vocab_size)
    for n in range(length):
        z = sample_index(theta)
        w = sample_index(word_dist[z,:])
        v[w] += 1
    return v

def gen_documents(word_dist, n_topics=10, vocab_size=25, n=1000, length=100, alpha=1.0):
    """
    Generate a document-term matrix.
    """
    m = np.zeros((n, vocab_size))
    for i in xrange(n):
        m[i, :] = gen_document(word_dist, n_topics, vocab_size, length, alpha)
    return m

def gen_lda2(T, W, D, N, alpha):
    phi = gen_word_distribution(T, N)
    matrix = gen_documents(phi, T, W, D, N, alpha)
    return matrix, phi

def write_stan(data_dir, idx, W, T, D, alpha, beta, B, chains=1, write_params=True):
    model_file = os.path.join(data_dir, 'lda.stan')
    with open(model_file, 'w') as fout:
        fout.write('''
data {
    int<lower=2> K; // num topics
    int<lower=2> V; // num words
    int<lower=1> M; // num docs
    int<lower=1> N; // total word instances
    int<lower=1,upper=V> w[N]; // word n
    int<lower=1,upper=M> doc[N]; // doc ID for word n
    vector<lower=0>[K] alpha; // topic prior
    vector<lower=0>[V] beta; // word prior
}

parameters {
    simplex[K] theta[M]; // topic dist for doc m
    simplex[V] phi[K]; // word dist for topic k
}

model {
    for (m in 1:M)
        theta[m] ~ dirichlet(alpha); // prior
    for (k in 1:K)
        phi[k] ~ dirichlet(beta); // prior
    for (n in 1:N) {
        real gamma[K];
        for (k in 1:K)
            gamma[k] <- log(theta[doc[n],k]) + log(phi[k,w[n]]);
        increment_log_prob(log_sum_exp(gamma)); // likelihood
    }
}''')
    fname = os.path.join(data_dir, 'lda_stan_{}.py'.format(idx))
    N = 0
    alpha_l = [alpha]*T
    beta_l = [beta]*W
    w = []
    doc = []
    for i,d in enumerate(B):
        for j,count in enumerate(d):
            if count>0:
                w += [j+1]*count
                N += count
                doc += [i+1]*count
    w_str = '['+','.join([str(i) for i in w])+']'
    doc_str = '['+','.join([str(i) for i in doc])+']'
    with open(fname, 'w') as fout:
        fout.write('''
import pystan
import numpy as np
''')
        if write_params:
            fout.write('''
lda_dat = {{
    'K' : {},
    'V' : {},
    'M' : {},
    'N' : {},
    'w' : {},
    'doc' : {},
    'alpha' : {},
    'beta' : {}
}}
'''.format(T, W, D, int(N), w_str, doc_str, alpha_l, beta_l))
        else:
            fout.write('''
lda_dat = {{
    'V' : {},
    'M' : {},
    'N' : {},
    'w' : {},
    'doc' : {}
}}
'''.format(W, D, int(N), w_str, doc_str))
        fout.write('''
fit = pystan.stan(file='{}', data=lda_dat, chains={}, **kwargs)
np.savez(out_file, fit.get_logposterior()[0])
np.savez(phi_file, fit['phi'][-1,:,:])
np.savez(theta_file, fit['theta'][-1,:,:])
'''.format(model_file, chains))

def write_prism(data_dir, idx, W, T, D, alpha, beta, B, write_params=True):
    fname = os.path.join(data_dir, 'lda_prism_{}.psm'.format(idx))
    #theta_str = '\n'.join(['values(theta({}), [1-{}], a@{}).'.format(i+1,T,alpha)
    #    for i in range(D)])
    #phi_str = '\n'.join(['values(phi({}), [1-{}], a@{}).'.format(i+1,W,beta)
    #    for i in range(T)])
    theta_str = 'values(theta(_), [1-{}], a@{}).'.format(T,alpha)
    phi_str = 'values(phi(_), [1-{}], a@{}).'.format(W,beta)
    with open(fname, 'w') as fout:
        fout.write('''% LDA

''')
        if write_params:
            fout.write('''

{}

{}
'''.format(theta_str, phi_str))
        fout.write('''generate(W, D) :-
    msw(theta(D), T),
    msw(phi(T), W).

repl(X, N, L) :-
    length(L, N),
    maplist(Y, Y = X, L).

goal_list(GL) :-
    observe(d(D),L),
    member((w(W),C), L),
    repl((W,D), C, GL).

go :-
    findall(generate(W,D), (goal_list(GL), member((W,D), GL)), Goals),
    learn_b(Goals).

%prism_main :-
%    set_prism_flag(learn_mode,both),
%    go,
%    save_phi.
    %save_phi_d,
    %save_theta.

%save_phi :-
%    findall(Param, get_sw(phi(_), [_,_,Param]), Params),
%    open('data/out_lda_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

%save_phi_d :-
%    findall(Param, get_sw_d(phi(_), [_,_,Param]), Params),
%    open('data/out_d_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

%save_theta :-
%    findall(Param, get_sw(theta(_), [_,_,Param]), Params),
%    open('data/out_theta_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

''')
        for i,doc in enumerate(B):
            word_list = []
            for j,count in enumerate(doc):
                if count>0:
                    word_list.append('(w({}) , {})'.format(j+1, int(count)))
            str_word_list = '['+','.join(word_list)+']'
            fout.write('observe(d({}), {}).\n'.format(i+1, str_word_list))

def write_pb_cv(data_dir, idx, W, T, D, alpha, beta, Bs, write_params=True):
    for (k,counts) in enumerate(Bs):
        fname = os.path.join(data_dir, 'lda_{}_{}.pb'.format(idx, k))
        with open(fname, 'w') as fout:
            if write_params:
                fout.write('''% LDA artificial data

% needed to ground constraints
:- enforce_labeling(true).

pb_dirichlet({}, mu, {}, {}).
pb_dirichlet({}, phi, {}, {}).

generate(Doc, Token) :-
    Topic in 1..{},
    mu(Topic,Doc),
    phi(Token,Topic).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [generate(Doc, Token)]
).

'''.format(alpha, T, D, beta, W, T, T))
            else:
                fout.write('''% LDA artificial data

% needed to ground constraints
:- enforce_labeling(true).

pb_plate(
    [observe(d(Doc), TokenList), member((w(Token), Count), TokenList)],
    Count,
    [generate(Doc, Token)]
).

''')
            for d in range(counts.shape[0]):
                out_str = 'observe(d({}), ['.format(d+1)
                list_str = []
                crow = counts[d,:].tocoo()
                for term_idx,count in itertools.izip(crow.col, crow.data):
                    list_str.append('(w({}),{})'.format(term_idx+1, int(count)))
                out_str += ','.join(list_str)
                out_str += ']).\n'
                fout.write(out_str)

def write_stan_cv(data_dir, idx, W, T, D, alpha, beta, Bs, chains=1, write_params=True):
    model_file = os.path.join(data_dir, 'lda.stan')
    with open(model_file, 'w') as fout:
        fout.write('''
data {
    int<lower=2> K; // num topics
    int<lower=2> V; // num words
    int<lower=1> M; // num docs
    int<lower=1> N; // total word instances
    int<lower=1,upper=V> w[N]; // word n
    int<lower=1,upper=M> doc[N]; // doc ID for word n
    vector<lower=0>[K] alpha; // topic prior
    vector<lower=0>[V] beta; // word prior
}

parameters {
    simplex[K] theta[M]; // topic dist for doc m
    simplex[V] phi[K]; // word dist for topic k
}

model {
    for (m in 1:M)
        theta[m] ~ dirichlet(alpha); // prior
    for (k in 1:K)
        phi[k] ~ dirichlet(beta); // prior
    for (n in 1:N) {
        real gamma[K];
        for (k in 1:K)
            gamma[k] <- log(theta[doc[n],k]) + log(phi[k,w[n]]);
        increment_log_prob(log_sum_exp(gamma)); // likelihood
    }
}''')
    for (k,counts) in enumerate(Bs):
        fname = os.path.join(data_dir, 'lda_stan_{}_{}.py'.format(idx,k))
        N = 0
        alpha_l = [alpha]*T
        beta_l = [beta]*W
        w = []
        doc = []
        for d in range(counts.shape[0]):
            crow = counts[d,:].tocoo()
            for term_idx,count in itertools.izip(crow.col, crow.data):
                if count>0:
                    w += [term_idx+1]*count
                    N += count
                    doc += [d+1]*count
        w_str = '['+','.join([str(i) for i in w])+']'
        doc_str = '['+','.join([str(i) for i in doc])+']'
        with open(fname, 'w') as fout:
            fout.write('''
import pystan
import numpy as np
''')
            if write_params:
                fout.write('''
lda_dat = {{
    'K' : {},
    'V' : {},
    'M' : {},
    'N' : {},
    'w' : {},
    'doc' : {},
    'alpha' : {},
    'beta' : {}
}}
    '''.format(T, W, D, int(N), w_str, doc_str, alpha_l, beta_l))
            else:
                fout.write('''
lda_dat = {{
    'V' : {},
    'M' : {},
    'N' : {},
    'w' : {},
    'doc' : {}
}}
    '''.format(W, D, int(N), w_str, doc_str))
            fout.write('''
fit = pystan.stan(file='{}', data=lda_dat, chains={}, **kwargs)
np.savez(out_file, fit.get_logposterior()[0])
np.savez(phi_file, fit['phi'][-1,:,:])
np.savez(theta_file, fit['theta'][-1,:,:])
    '''.format(model_file, chains))

def write_prism_cv(data_dir, idx, W, T, D, alpha, beta, Bs, write_params=True):
    for (k,counts) in enumerate(Bs):
        fname = os.path.join(data_dir, 'lda_prism_{}_{}.psm'.format(idx, k))
        theta_str = 'values(theta(_), [1-{}], a@{}).'.format(T,alpha)
        phi_str = 'values(phi(_), [1-{}], a@{}).'.format(W,beta)
        with open(fname, 'w') as fout:
            fout.write('''% LDA

    ''')
            if write_params:
                fout.write('''

{}

{}
    '''.format(theta_str, phi_str))
            fout.write('''generate(W, D) :-
    msw(theta(D), T),
    msw(phi(T), W).

repl(X, N, L) :-
    length(L, N),
    maplist(Y, Y = X, L).

goal_list(GL) :-
    observe(d(D),L),
    member((w(W),C), L),
    repl((W,D), C, GL).

go :-
    findall(generate(W,D), (goal_list(GL), member((W,D), GL)), Goals),
    learn_b(Goals).

    %prism_main :-
    %    set_prism_flag(learn_mode,both),
    %    go,
    %    save_phi.
        %save_phi_d,
        %save_theta.

    %save_phi :-
    %    findall(Param, get_sw(phi(_), [_,_,Param]), Params),
    %    open('data/out_lda_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

    %save_phi_d :-
    %    findall(Param, get_sw_d(phi(_), [_,_,Param]), Params),
    %    open('data/out_d_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

    %save_theta :-
    %    findall(Param, get_sw(theta(_), [_,_,Param]), Params),
    %    open('data/out_theta_prism',write, Stream), forall(member(Param, Params), (write(Stream,Param), nl(Stream))), close(Stream).

    ''')
            for d in range(counts.shape[0]):
                out_str = 'observe(d({}), ['.format(d+1)
                list_str = []
                crow = counts[d,:].tocoo()
                for term_idx,count in itertools.izip(crow.col, crow.data):
                    list_str.append('(w({}),{})'.format(term_idx+1, int(count)))
                out_str += ','.join(list_str)
                out_str += ']).\n'
                fout.write(out_str)

def write_txt_cv(data_dir, idx, Bs, T, alpha, beta, write_params=True):
    for (k,counts) in enumerate(Bs):
        fname2 = os.path.join(data_dir, 'lda_{}_{}.txt'.format(idx,k))
        with open(fname2, 'w') as fout:
            cx = counts.tocoo()
            prev_i = -1
            for i,j,v in itertools.izip(cx.row, cx.col, cx.data):
                if not (i-1 == prev_i or i == prev_i):
                    print('no words in document: {}'.format(i-1))
                prev_i = i
                fout.write('{} {} {}\n'.format(i+1, j+1, v))
            #for i,doc in enumerate(B):
            #    for j,count in enumerate(doc):
            #        if count>0:
            #            fout.write('{} {} {}\n'.format(i+1, j+1, count))
    with open(os.path.join(data_dir, 'gibbs_lda.R'), 'w') as fout:
        fout.write('''
# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)
''')
        if write_params:
            fout.write('''
K = {}
alpha = {}
beta = {}
# iter = ...
# seed = 
'''.format(T, alpha, beta))
        fout.write('''set.seed(seed)
run_gibbs <- function(f){{
  print(f)
  data = read.table(f, quote="\\\"")
  lapply(sample(1:5000, 1, replace=F), function(s){{
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="Gibbs", control=list(
      seed = s,
      alpha = alpha,
      delta = beta, # this is beta
      keep = 1, # save log likelihood every iteration
      iter = iter,
      burnin = 0
      ))
    model
    }})
}}        
''')    
    with open(os.path.join(data_dir, 'vem_lda.R'), 'w') as fout:
        fout.write('''
# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)
''')
        if write_params:
            fout.write('''
K = {}
alpha = {}
beta = {}
# iter = ...
# seed = 
'''.format(T, alpha, beta))
        fout.write('''set.seed(seed)
run_vem <- function(f){{
  print(f)
  data = read.table(f, quote="\\\"")
  lapply(sample(1:5000, 1, replace=F), function(s){{
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="VEM", control=list(
        seed = seed,
        alpha = alpha,
        em = list(iter.max=iter),
        keep = 1 # save log likelihood every iteration
  ))
  model
  }})
}}        
''')                       


