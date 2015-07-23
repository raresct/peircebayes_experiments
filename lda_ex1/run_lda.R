# install packages if necessary
list.of.packages <- c("topicmodels", "tm", "slam")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# load packages
lapply(list.of.packages, library, character.only=T)

K = 10

set.seed(1234)
run_gibbs <- function(f){
  print(f)
  data = read.table(f, quote="\"")
  lapply(sample(1:5000, 10, replace=F), function(s){
    model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="Gibbs", control=list(
      seed = s,
      alpha = 1.,
      delta = 1., # this is beta
      keep = 1, # save log likelihood every iteration
      iter = 200,
      burnin = 0
      ))
    model@logLiks
    })
}
#Rprof(filename="data/RProf.out")
gibbs_lls = lapply(list.files('data', pattern='*.txt', full.names=T), run_gibbs)
write.table(gibbs_lls, "data/gibbs_ll", sep=" ", row.names=F, col.names=F)
#Rprof(tmp <- tempfile())
#prof_summ <- summaryRprof(filename="data/RProf.out")
#write.table(prof_summ$sampling.time, "data/time_r", sep=" ", row.names=F, col.names=F)

# run_vem <- function(f){
#   print(f)
#   data = read.table(f, quote="\"")
#   model = LDA(simple_triplet_matrix(data[,1], data[,2], data[,3]), K, method="VEM", control=list(
#     seed = 1234,
#     alpha = 1.,
#     em = list(iter.max=150),
#     keep = 1 # save log likelihood every iteration
#   ))
#   model@logLiks
# }
# 
# vem_lls = lapply(list.files('data', pattern='*.txt', full.names=T), run_vem)
# min_ll = min(sapply(vem_lls, length))
# tr_vem_lls = lapply(vem_lls, function(x) x[1:min_ll])
# write.table(tr_vem_lls, "data/vem_ll", sep=" ", row.names=F, col.names=F)

