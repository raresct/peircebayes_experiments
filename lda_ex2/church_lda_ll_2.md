(define (index-in x xs)
  (define (loop x k rst)
    (if (is_null rst) k
      (if (equal? (first rst) x) k
        (loop x (+ k 1) (rest rst)))))
    (loop x 0 xs))

(define (word-factor i distr) 
  (let ((p (list-ref distr i)))
    (factor (log p))))

(define number-of-topics 2)

(define vocabulary '("A" "B" "C" "D"))

(define documents
  '(("A" "A" "A") ("A" "C" "A") ("C" "C" "A")
    ("A" "A" "A") ("A" "C" "A") ("C" "C" "A")
    ("A" "A" "A") ("A" "C" "A") ("C" "C" "A")
    ("A" "A" "A") ("A" "C" "A") ("C" "C" "A")
    ("D" "D" "D") ("B" "D" "B") ("B" "B" "D")))

(define samples
  (mh-query    
   100 10   
   
   (define topic-word-distributions
     (repeat number-of-topics 
             (lambda () (dirichlet '(0.4 0.4 0.4 0.4)))))
   
   ((define process
     (sum (map
      (lambda (document)
        (let* ((topic-selection-distr (dirichlet '( 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.))))
          (sum (map (lambda (word)
                 (let* ((sampled-topic (multinomial topics topic-selection-distr))
                        (idx (index-in word vocabulary)))
                   (+ (log (list-ref (list-ref topic-word-distributions sampled-topic) idx))
                      (log (list-ref topic-selection-distr sampled-topic)))    
                   ))
          document))))
      documents)))
   process
   #t))

; for web
;(lineplot (map (lambda (x) (list x (list-ref samples x))) (range 0 (- (length samples) 1))))

; for command line
(write-csv (list samples) "church2_ll.csv"" " ")
