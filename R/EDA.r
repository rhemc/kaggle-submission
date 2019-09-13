require(gplots)
require(RColorBrewer)
require(ClustOfVar)
require(FactoMineR)
require(plyr)
require(gbm)
require(randomForest)
if(getwd() != "/home/kurt/Desktop/sta314/data"){
  setwd("/home/kurt/Desktop/sta314/data")
  data = read.csv("trainingdata.csv")
  predict_these = read.csv("test_predictors.csv")
}
if(getwd() != "/home/kurt/Desktop/sta314/R"){
  setwd("/home/kurt/Desktop/sta314/R")
}

normalize <- function(data){
  return(data-mean(data))/sd(data)
}


hists <- function(data){
  for (i in 2:ncol(data)) {
    hist(data[,i],main=names(data)[i]) 
  }
}

partial_plots <- function(data){
  for (i in 2:ncol(data)) {
    plot(data[,i],data$y,main = names(data)[i]) 
  }
}

ks_matrix <- function(data){
  len = ncol(data)
  pvals <- matrix( rep( 0, len=(len-1)^2), nrow = (len-1))
  for (i in 2:len) {
    for (j in 2:len){
      pvals[i-1,j-1] = ks.test(data[,i],data[,j])$p.value
    }
  }
  return(pvals)
}

plot_heat_map <- function(mat){
  rownames(mat) <- names(data)[-1]
  my_palette <- colorRampPalette(c("red", "yellow", "green"))(n = 299)
  heatmap.2(mat,dendrogram='none', Rowv=TRUE, Colv=TRUE,main="Pvalues of KS test",col=my_palette)
}

pca <- function(data){
  data$y <- NULL
  properties <- FAMD(data,ncp=ncol(data),graph=F)
  print(properties$eig)
  plot(properties$eig[,3])
  return(as.data.frame(properties$ind$coord))
}


variable_importance_boost <- function(data){
  boosted_tree = gbm(y~.,
                     data=data,
                     distribution='gaussian',
                     n.trees = 5000,
                     interaction.depth = 6,
                     shrinkage = 0.01,
                     cv = 5
  )
  best = gbm.perf(boosted_tree,method="cv")
  boosted_tree = gbm(y~.,
                     data=data,
                     distribution='gaussian',
                     n.trees = best,
                     interaction.depth = 6,
                     shrinkage = 0.01
  )
  summary(boosted_tree,main='variable importance from boosted tree',las=2)
  print("The important variables from BT")
  print(summary(boosted_tree))
}
variable_importance_rf <- function(data){
  rf = randomForest(y~.,data = data, mtry = (ncol(data)-1)/3, importance = TRUE)
  print("the important predictors from RF")
  print(data.frame(sort(importance(rf)[,1],decreasing=T)))
  varImpPlot(rf,main="variable importance from random forest")
}



#plot_heat_map(ks_matrix(data))
#data[,-1] <- sapply(data[,-1],normalize)
#factors = c("X11","X15","X17","X12","X13")
#data[,factors] <- lapply(round(data[,factors]),FUN=factor)
#hists(predict_these)
#partial_plots(data)
#pca_data <- pca(data)
#pca_data$y <- data$y
#variable_importance_boost(data)
#variable_importance_rf(data)
#variable_importance_boost(pca_data)
#variable_importance_rf(pca_data)
full_data <- cbind(data,read.csv('stack_predictors_train.csv'))
full_predict_these <- cbind(data,read.csv('stack_predictors_test.csv'))
variable_importance_boost(full_data)