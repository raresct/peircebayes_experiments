#!/usr/bin/env python2

import sys

sys.path.append("..") # for utils
from utils import rewrite_dir, call_cmd

def read_data(fname):
    with open(fname, 'r') as fin:
        # first line = number of items
        N = int(fin.readline().strip().split()[0])
        # rest = permutations
        data = [ [int(i) for i in line.strip().split()[2:]] 
            for line in fin] 
        return N, data

def write_pb(fname, N, data, K, beta, gamma):
    with open(fname, 'w') as fout:
        fout.write('''
% Sushi dataset

% needed to ground constraints
:- enforce_labeling(true).

''')
        fout.write('''
% prob distribs
pb_dirichlet({}, pi, {}, 1).
'''.format(beta, K))
        fout.write('\n'.join(
            ['pb_dirichlet({}, p{}, {}, {}).'.format(gamma, i, i, K)  
                for i in range(2, N+1)]))
        fout.write('''
        
% plate
pb_plate(
    [observe(Sample)],
    1,
    [generate({}, Sample)]
).
'''.format('['+','.join([str(i) for i in range(N)])+']'))
        fout.write('''
insert_rim([], ToIns, Ins,
    Pos, Ins1) :-
    append(Ins, [ToIns], Ins1),
    length(Ins1, Pos).
insert_rim([H|_T], ToIns, Ins,
    Pos, Ins1) :-
    nth1(Pos, Ins, H),
    nth1(Pos, Ins1, ToIns, Ins).
insert_rim([H|T] , ToIns, Ins,
    Pos, Ins1) :-
    \+member(H, Ins),
    insert_rim(T, ToIns, Ins,
        Pos, Ins1).

generate([H|T], Sample):-
    K in 1..{},
    pi(K,1),
    generate(T, Sample, [H], 2, K).

generate([], Sample, Sample, _Idx, _K).
generate([ToIns|T], Sample, Ins, Idx, K) :-
    % insert next element at Pos
    % yielding a new list Ins1
    append(_, [ToIns|Rest], Sample),
    insert_rim(Rest, ToIns, Ins,
        Pos, Ins1),
    % build prob predicate in Pred
    number_chars(Idx, LIdx),
    append(['p'], LIdx, LF),
    atom_chars(F, LF),
    Pred =.. [F, Pos, K],
    % call prob predicate
    pb_call(Pred),
    Idx1 is Idx+1,
    % recurse
    generate(T, Sample, Ins1, Idx1, K).
    
'''.format(K))
        fout.write('\n'.join(['observe(['+
            ','.join([str(i) for i in perm])+']).' 
            for perm in data]))
    print 'Done writing pb.'


def data_gen():
    K = 6
    beta = 50./float(K)
    gamma = 0.1
    fin_name = 'data/sushi3/sushi3a.5000.10.order'
    fout_name = 'data/sushi3/sushi3.pb'
    
    rewrite_dir('data')
    call_cmd('sh get_data.sh')
    N, data = read_data(fin_name)
    write_pb(fout_name, N, data, K, beta, gamma)

def main():
    data_gen()

if __name__=='__main__':
    main()

