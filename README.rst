
=================
2016
=================

1. 5.1 LDA.

.. code-block:: sh

    cd lda_artif1
    python2 data_gen.py
    python2 run_experiment.py
    python2 post_process.py
    # the plot is in data/lls.pdf

2. 5.2 LDA. $DIR is one of lda_ll_time, lda_ll_topics, lda_ull_time, lda_ull_topics, lda_p_time, lda_p_topics
    
.. code-block:: sh

    cd $DIR
    python2 data_gen.py
    # change path in the do_pb function in run_experiment.py
    python2 run_experiment.py
    jupyter notebook post_process.ipynb
    # Cell -> Run All
    # the plot is in data/lls.pdf

3. 5.3 Seed LDA. 

.. code-block:: sh

    cd seed_lda
    python2 run_clda.py
    # the plots are in data/topic_1.pdf and data/topic_2.pdf

4. 5.4 Cluster LDA.

.. code-block:: sh

    mv data_2 data
    python2 data_gen.py
    python2 run_experiment.py
    jupyter notebook post_proc.ipynb
    # Cell -> Run All
    # the plot is data/heat_fields.pdf

5. 5.5 RIM.

.. code-block:: sh

    cd rim_ex3
    python run_rim3.py
    # data/table.tex should look similar to Table 1 in the paper

=================
PLP 2015
=================

1. ex1

.. code-block:: sh

    cd lda_ex1
    python run_lda1.py
    # data/lls.pdf should look similar to Figure 3 in the paper

2. ex2

.. code-block:: sh

    cd rim_ex3
    python run_rim3.py
    # data/table.tex should look similar to Table 1 in the paper

3. ex3

.. code-block:: sh

    cd lda_ex2
    python run_lda2.py
    # data/lls.pdf should look similar to Figure 4 in the paper

