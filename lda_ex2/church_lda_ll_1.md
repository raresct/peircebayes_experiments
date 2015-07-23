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
(define vocabulary (append '(bear wolf)'(python prolog)))

(define topics '(0 1))

(define doc-length 10)

(define doc1 '(bear wolf bear wolf bear wolf python wolf bear wolf))
(define doc2 '(python prolog python prolog python prolog python prolog python prolog))
(define doc3 '(bear wolf bear wolf bear wolf bear wolf bear wolf))
(define doc4 '(python prolog python prolog python prolog python prolog python prolog))
(define doc5 '(bear wolf bear python bear wolf bear wolf bear wolf))

(define docs (list doc1 doc2 doc3 doc4 doc5))
;(display docs)

(define doc->wid (lambda (word)
  (list-index vocabulary word))  
)

(define docs->wid 
  (lambda (doc) (map doc->wid doc)))

(define w 
  (map docs->wid docs))

; this is w
;(display w)



(define samples
  (mh-query
   100 10
   
   (define document->mixture-params
     (mem (lambda (doc-id) (dirichlet (make-list (length topics) 1.0)))))
   
   (define topic->mixture-params
     (mem (lambda (topic) (dirichlet (make-list (length vocabulary) 0.1)))))
   
   (define document->topics
     (mem (lambda (doc-id)
            (repeat doc-length
                    (lambda () (multinomial topics (document->mixture-params doc-id)))))))
   
   (define document->words
     (mem (lambda (doc-id)
            (map (lambda (topic)
                   (multinomial vocabulary (topic->mixture-params topic)))
                 (document->topics doc-id)))))
   
   ;(map topic->mixture-params topics)
   ; this is z
   (define z 
     (map document->topics '(doc1 doc2 doc3 doc4 doc5)))
   
   ; ll per document
   (define (doc-ll doc-w doc-z doc-idx)
     (if (and (null? doc-w) (null? doc-z))
       0
       (+ (log (list-ref (topic->mixture-params (first doc-z)) (first doc-w))) ; phi
          (log (list-ref (document->mixture-params doc-idx) (first doc-z)))    ; theta
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
    (factor-equal? (document->words 'doc1) doc1)
    (factor-equal? (document->words 'doc2) doc2)
    (factor-equal? (document->words 'doc3) doc3)
    (factor-equal? (document->words 'doc4) doc4)
    (factor-equal? (document->words 'doc5) doc5))))

; for web
(lineplot (map (lambda (x) (list x (list-ref samples x))) (range 0 (- (length samples) 1))))

; for command line 
;(write-csv (list samples) "church1_ll.csv"" " ")

