#!/usr/bin/env Rscript
# HCPC consolidation, with the Ward cut supplied by the Python vector routine.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L, !file.exists(args[2]), !file.exists(args[3]))
x <- read.csv(args[1], check.names=FALSE, stringsAsFactors=FALSE)
key <- x$person_wave_key
initial <- x$initial_cluster
z <- as.matrix(x[,grep("^Dim[._]",names(x)),drop=FALSE])
stopifnot(ncol(z)==5L, all(is.finite(z)), !anyDuplicated(key),
          identical(sort(unique(initial)),1:2))
centers <- do.call(rbind,lapply(1:2,function(k)colMeans(z[initial==k,,drop=FALSE])))
set.seed(20260901L)
fit <- stats::kmeans(z,centers=centers,iter.max=10L)
stopifnot(fit$ifault==0L, all(fit$size>0), all(is.finite(fit$centers)))
ordering <- order(fit$centers[,1])
labels <- order(ordering)[fit$cluster]
write.csv(data.frame(person_wave_key=key,cluster=labels),args[2],row.names=FALSE)
jsonlite::write_json(list(algorithm="R stats::kmeans Hartigan-Wong",iter_max=10L,
                         iterations=fit$iter,ifault=fit$ifault,
                         within_ss=fit$tot.withinss,R_version=R.version.string),
                    args[3],auto_unbox=TRUE,pretty=TRUE)
Sys.chmod(args[2:3],"0600")
