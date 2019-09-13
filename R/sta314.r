require(caret)
require(glmnet)
require(nnet)
require(tree)
require(rpart)
require(xgboost)

require(functional)
require(gsubfn)
require(parallel)

#need to check if we are not in this folder, otherwise setwd will throw an error
if(!grepl(getwd(),"/data")){
  setwd("../data")
  data = read.csv("trainingdata.csv")
  predict_these = read.csv("test_predictors.csv")
}
if(!grepl(getwd(),"/R")){
  setwd("../R")
}

normalize <- function(data){
  return(data-mean(data))/sd(data)
}

rmse <- function(actual,predicted){
  if(length(actual) != length(predicted)){
    stop("rmse actual != predicted")
  }
  sqrt(sum(actual-predicted)^2/length(actual))
}

partition_data <- function(data,split_percent){
  train_index = createDataPartition(data$y,p=split_percent, list=FALSE)
  train_data = data[train_index,]
  test_data = data[-train_index,]
  return(list(train_data=train_data,test_data=test_data))
}

k_fold_partition <- function(data,k){
  n = nrow(data)
  fold_size = floor(n/k)
  indicies <- sample(1:n,n)
  partitions = list()
  i <- 1
  take <- 0
  while(take < n){
    partitions[[i]] <- data[indicies[(take+1):(take+fold_size)],]
    take <- take + fold_size
    i<- i + 1
  }
  return(partitions)
}

linear_prediction <- function(train_data,test_data,hparam=F){
  lin_model <- lm(y~.,data=train_data)
  if(class(test_data) == "logical"){
    return(lin_model)
  }
  err <- rmse(predict(lin_model,test_data),test_data[,1])
  return(err)
}

lasso_prediction <- function(train_data,test_data,lambda){
  x_train <- as.matrix(train_data[,-1])
  lasso_model <- glmnet(x_train, train_data[,1], alpha = 1, lambda = lambda)
  if(class(test_data) == "logical"){
    return(lasso_model)
  }
  x_test <- as.matrix(test_data[,-1])
  err <- rmse(predict(lasso_model,x_test),test_data[,1])
  return(err)
}

ridge_prediction <- function(train_data,test_data,lambda){
  x_train <- as.matrix(train_data[,-1])
  ridge_model <- glmnet(x_train, train_data[,1], alpha = 0, lambda = lambda)
  if(class(test_data) == "logical"){
    return(ridge_model)
  }
  x_test <- as.matrix(test_data[,-1])
  return(rmse(predict(ridge_model,x_test),test_data[,1]))
  
}

nn_prediction <- function(train_data,test_data,hparam){
  nn_model <- nnet(y~.,data=train_data,linout=TRUE,size=floor(hparam),maxit=100,trace=F)
  if(class(test_data) == "logical"){
    return(nn_model)
  }
  return(rmse(predict(nn_model,test_data[,-1]),test_data[,1]))
}

tree_prediction <- function(train_data,test_data,hparam){
  hparam <- floor(hparam)
  if(hparam <= 1){
    stop("dtree size <= 1")
  }
  tree_model = tree(y~.,data=train_data)
  tree_model = prune.tree(tree_model,best=hparam)
  if(class(test_data) == "logical"){
    return(tree_model)
  }
  return(rmse(predict(tree_model,test_data[,-1]),test_data[,1]))
}

knn_prediction <- function(train_data,test_data,K){
  x_train <- as.matrix(train_data[,-1])
  knn_model <- knnreg(x_train, train_data[,1], k=K)
  if(class(test_data) == "logical"){
    return(knn_model)
  }
  x_test <- as.matrix(test_data[,-1])
  return(rmse(predict(knn_model,x_test),test_data[,1]))

}

predict_safe <- function(model,data){
  if(class(model) != "elnet" || class(model) != "knnreg"){#tree models break try catch for some mysterious reason
    data$y = NULL
    return(predict(model,data))
  } else {
    data$y = NULL
    return(predict(model,as.matrix(data))) 
  }
}

predict_bag <- function(model,data){
  if(class(model) != "list"){
    return(predict_safe(model,data))
  }
  n <- length(model)
  sum_prediction <- array(0,nrow(data))
  predictions <- mclapply(model,function(i) { predict_safe(i,data) })
  for(i in 1:length(predictions)){
    sum_prediction <- sum_prediction + as.vector(predictions[[i]])
  }
  sum_prediction <- sum_prediction / n #average the predictions
  return(sum_prediction)
}

bagged_model <- function(train_data,test_data,hparam,model_function,number_of_samples){
  bag_model = list()
  for(i in 1:number_of_samples){
  samp <- train_data[sample(nrow(train_data),nrow(train_data),replace=T),]
  bag_model[[i]] <- model_function(samp,F,hparam)
  }
  if(class(test_data) == "logical"){
    return(bag_model)
  }
  return(rmse(predict_bag(bag_model,test_data[,-1]),test_data[,1]))
}

turn_into_csv <- function(data,name='Submission.csv'){
  da.sample = data.frame(cbind(1:500,data))
  names(da.sample) = c('id','y') 
  write.csv(da.sample,file=name,row.names=FALSE)
}

k_fold_model <- function(data,k,hparam_vals,model_function,iters,model_name){
  rmse_list <- list()
  for(l in 1:iters){
    print(paste("iteration",l))
    partitions <- k_fold_partition(data,k)
    for(i in 1:length(partitions)){
      test_data <- partitions[[i]]
      train_data <- data.frame()
      for(j in 1:length(partitions)){
        if(j != i){
          train_data <- rbind(train_data,partitions[[j]])
        }
      }
      for(m in 1:length(hparam_vals)){
        hparam <- hparam_vals[m]
        err <- model_function(train_data,test_data,hparam)
        if(length(rmse_list) >= m){
          rmse_list[[m]] <- append(rmse_list[[m]],err)
        }else{  
          rmse_list <- append(rmse_list,err)
        }
      }
    }
  }
  boxplot(rmse_list,names=hparam_vals,las=2,main=model_name,xlab='Hyper Parameter Value',ylab='RMSE')
  agg <- sapply(rmse_list,mean)
  print(paste('-----',model_name,'-----'))
  print("RMSE list corresponding to hyperparameters:")
  print(agg)
  best_rmse = min(agg)
  print(paste("minimum RMSE",best_rmse))
  best_hparam = hparam_vals[which.min(agg)]
  print(paste("best hyperparameter",best_hparam))
  return(list(hparam=best_hparam,rmse=best_rmse))
}

explore_models <- function(data){
  
  #lm
  print('building lm')
  list[lm_hparam,lm_rmse] = k_fold_model(data,5,0,linear_prediction,10,'Linear Regression')
  
  #lasso
  print('building lasso')
  lasso_hparam_vals = seq(0, 0.1, length.out = 11)
  list[lasso_hparam,lasso_rmse] = k_fold_model(data,5,lasso_hparam_vals,lasso_prediction,10,'Lasso')
  
  #ridge
  print('building ridge')
  ridge_hparam_vals = seq(0, 0.1, length.out = 11)
  list[ridge_hparam,ridge_rmse] = k_fold_model(data,5,ridge_hparam_vals,ridge_prediction,10,'Ridge Regression')
  
  #knn
  print('building knn')
  knn_hparam_vals <- seq(1,25,by=1)
  list[knn_hparam,knn_rmse] = k_fold_model(data,5,knn_hparam_vals,knn_prediction,10,'K Nearest Neighbours')
  
  #trees
  print('building tree')
  tree_hparam_vals <- seq(2,10,by=1)
  list[tree_hparam,tree_rmse] = k_fold_model(data,5,tree_hparam_vals,tree_prediction,10,'Decision Tree')
  
  #nn
  print('building nn')
  nn_hparam_vals <- seq(2,10,by=1)
  list[nn_hparam,nn_rmse] = k_fold_model(data,5,nn_hparam_vals,nn_prediction,5,'Neural Net')
  
  #bagged lm
  print('building bagged lm')
  lm_bag_prediction <- Curry(bagged_model,model_function=linear_prediction,number_of_samples=50)
  list[lm_hparam,lm_rmse] = k_fold_model(data,5,0,lm_bag_prediction,10,'Bagged Linear Regression')
  
  #bagged knn
  print('building bagged knn')
  knn_bag_prediction <- Curry(bagged_model,model_function=knn_prediction,number_of_samples=50)
  list[bag_knn_hparam,bag_knn_rmse] = k_fold_model(data,5,knn_hparam_vals,knn_bag_prediction,5,'Bagged K Nearest Neighbours')
  
  #bagged trees
  print('building bagged tree')
  tree_bag_prediction <- Curry(bagged_model,model_function=tree_prediction,number_of_samples=50)
  list[bag_tree_hparam,bag_tree_rmse] = k_fold_model(data,5,tree_hparam_vals,tree_bag_prediction,5,'Bagged Trees')
  
  #bagged nn
  print('building bagged nn')
  nn_bag_prediction <- Curry(bagged_model,model_function=nn_prediction,number_of_samples=50)
  list[bag_nn_hparam,bag_nn_rmse] = k_fold_model(data,5,nn_hparam_vals,nn_bag_prediction,5,'Bagged Neural Nets')
  
  #info
  print(sprintf('RMSEs of lm %f, lasso %f, ridge %f, knn %f, bag_knn %f, tree %f, bag_tree %f, nn %f, bag_nn %f',
                lm_rmse,lasso_rmse,ridge_rmse,knn_rmse,bag_knn_rmse,tree_rmse,bag_tree_rmse,nn_rmse,bag_knn_rmse))
  print(sprintf('Hyper Parameters of lasso %f, ridge %f, knn %f,bag_knn %f, tree %f, bag_tree %f, nn %f bag_nn %f',
                lasso_hparam,ridge_hparam,knn_hparam,bag_knn_hparam,tree_hparam,bag_tree_hparam,nn_hparam,bag_nn_hparam))
  
  #same_level
  rmse_list <- list()
  for(i in 1:50){
    list[train_data,test_data] = partition_data(data,0.5)
    if(i == 1){
      rmse_list[[1]] <- linear_prediction(train_data,test_data,0)
      rmse_list[[2]] <- lm_bag_prediction(train_data,test_data,0)
      rmse_list[[3]] <- lasso_prediction(train_data,test_data,lasso_hparam)
      rmse_list[[4]] <- ridge_prediction(train_data,test_data,ridge_hparam)
      rmse_list[[5]] <- knn_prediction(train_data,test_data,knn_hparam)
      rmse_list[[6]] <- knn_bag_prediction(train_data,test_data,bag_knn_hparam)
      rmse_list[[7]] <- tree_prediction(train_data,test_data,tree_hparam)
      rmse_list[[8]] <- tree_bag_prediction(train_data,test_data,bag_tree_hparam)
      rmse_list[[9]] <- nn_prediction(train_data,test_data,nn_hparam)
      rmse_list[[10]] <- nn_bag_prediction(train_data,test_data,bag_nn_hparam)
    }else{
      rmse_list[[1]] <- append(rmse_list[[1]],linear_prediction(train_data,test_data,0))
      rmse_list[[2]] <- append(rmse_list[[2]],lm_bag_prediction(train_data,test_data,0))
      rmse_list[[3]] <- append(rmse_list[[3]],lasso_prediction(train_data,test_data,lasso_hparam))
      rmse_list[[4]] <- append(rmse_list[[4]],ridge_prediction(train_data,test_data,ridge_hparam))
      rmse_list[[5]] <- append(rmse_list[[5]],knn_prediction(train_data,test_data,knn_hparam))
      rmse_list[[6]] <- append(rmse_list[[6]],knn_bag_prediction(train_data,test_data,bag_knn_hparam))
      rmse_list[[7]] <- append(rmse_list[[7]],tree_prediction(train_data,test_data,tree_hparam))
      rmse_list[[8]] <- append(rmse_list[[8]],tree_bag_prediction(train_data,test_data,bag_tree_hparam))
      rmse_list[[9]] <- append(rmse_list[[9]],nn_prediction(train_data,test_data,nn_hparam))
      rmse_list[[10]] <- append(rmse_list[[10]],nn_bag_prediction(train_data,test_data,bag_nn_hparam))
    }
  }
  name <- c('lm','bag lm','lasso','ridge','knn','bag knn','tree','bag tree','nn','bag nn')
  boxplot(rmse_list,names=name,las=2,Main='Best model (from K-fold cv) vs RMSE',xlab='Model Name',ylab='RMSE')
}

get_training_predictions <- function(partitions,model,hparam){
  predictions <- c()
  for(i in 1:length(partitions)){
    test_data <- partitions[[i]]
    train_data <- data.frame()
    for(j in 1:length(partitions)){
      if(j != i){
        train_data <- rbind(train_data,partitions[[j]])
      }
    }
    rnames <- rownames(test_data)
    test_predictions <- predict_bag(model(train_data,F,hparam),test_data)
    names(test_predictions) <- rnames
    predictions <- append(predictions,test_predictions)
  }
  return(predictions[order(as.numeric(names(predictions)))])
}

build_stack <- function(data,predict_these){
  predict_these$X1.500 <- NULL
  #lienar model
  print('linear model')
  #list[lm_hparam,lm_rmse] = k_fold_model(data,5,0,linear_prediction,10,'Linear Regression')
  lm_preds_train <- get_training_predictions(k_fold_partition(data,5),linear_prediction,0)
  lm_preds_test <- predict_safe(linear_prediction(data,F,0),predict_these)
  
  #nn size 10
  print('size 10 neural net')
  nn_preds_train <- get_training_predictions(k_fold_partition(data,5),nn_prediction,10)
  nn_preds_test = predict_safe(nn_prediction(data,F,10),predict_these)
  
  #nn size 6
  print('size 6 neural net')
  nn_preds_train2 <- get_training_predictions(k_fold_partition(data,5),nn_prediction,6)
  nn_preds_test2 = predict_safe(nn_prediction(data,F,6),predict_these)
  
  #knn k=2
  print('2nn')
  knn_preds_train <- get_training_predictions(k_fold_partition(data,5),knn_prediction,2)
  knn_preds_test = predict_safe(knn_prediction(data,F,2),predict_these)
  
  #knn k=7
  print('7nn')
  knn_preds_train2 <- get_training_predictions(k_fold_partition(data,5),knn_prediction,7)
  knn_preds_test2 = predict_safe(knn_prediction(data,F,7),predict_these)
  
  #bagged knn
  
  print('building bagged knn')
  #knn_hparam_vals <- seq(1,25,by=1)
  knn_bag_prediction <- Curry(bagged_model,model_function=knn_prediction,number_of_samples=1000)
  #list[bag_knn_hparam,bag_knn_rmse] = k_fold_model(data,5,knn_hparam_vals,knn_bag_prediction,5,'Bagged KNN (B=5000)') #best hparam 8
  bag_knn_preds_train <- get_training_predictions(k_fold_partition(data,5),knn_bag_prediction,8)
  bag_knn_preds_test = predict_bag(knn_bag_prediction(data,F,8),predict_these)
  
  #bagged trees
  print('building bagged tree')
  #tree_hparam_vals <- seq(2,15,by=1)
  tree_bag_prediction <- Curry(bagged_model,model_function=tree_prediction,number_of_samples=1000)
  #list[bag_tree_hparam,bag_tree_rmse] = k_fold_model(data,5,tree_hparam_vals,tree_bag_prediction,5,'Bagged Trees (B=5000)')
  bag_tree_preds_train <- get_training_predictions(k_fold_partition(data,5),tree_bag_prediction,5)
  bag_tree_preds_test = predict_bag(tree_bag_prediction(data,F,5),predict_these)
  
  #bagged nn
  print('building bagged nn')
  #nn_hparam_vals <- seq(2,15,by=1)
  nn_bag_prediction <- Curry(bagged_model,model_function=nn_prediction,number_of_samples=1000)
  #list[bag_nn_hparam,bag_nn_rmse] = k_fold_model(data,5,nn_hparam_vals,nn_bag_prediction,5,'Bagged NN (B=5000)')
  bag_nn_preds_train <- get_training_predictions(k_fold_partition(data,5),nn_bag_prediction,4)
  bag_nn_preds_test = predict_bag(nn_bag_prediction(data,F,4),predict_these)
  
  #write to file
  stack_train = data.frame(cbind(lm_preds_train,nn_preds_train,nn_preds_train2,knn_preds_train,knn_preds_train2,bag_knn_preds_train,bag_tree_preds_train,bag_nn_preds_train))
  stack_test = data.frame(cbind(lm_preds_test,nn_preds_test,nn_preds_test2,knn_preds_test,knn_preds_test2,bag_knn_preds_test,bag_tree_preds_test,bag_nn_preds_test))
  cnames <- c('lm','nn_size10','nn_size6','2nn','7nn','bag_knn','bag_tree','bag_nn')
  names(stack_train) <-  cnames
  names(stack_test) <- cnames
  write.csv(stack_train,file='restricted_stack_predictors_train.csv',row.names=FALSE)
  write.csv(stack_test,file='restricted_stack_predictors_test.csv',row.names=FALSE)
}

#temp <- data$y
#data <- sapply(data[,-1],normalize)
#data <- data.frame(y=temp,data)


##exploring models, and building stack

predictors <- c("X12","X23","X25","X8","X1","X4","X5","X2","X7")
temp <- data$y
data <-  data[,predictors]
data <- data.frame(y=temp,data)

pdf("explore_models.pdf")
print(system.time(explore_models(data)))
dev.off()

pdf("stack_components.pdf")
print(system.time(build_stack(data,predict_these[,predictors])))
dev.off()



##building final model

fit_model <- function(data,reps){
  list[train_data,test_data] = partition_data(data,0.5)
  
  cv_ctrl <- trainControl(method = "repeatedcv", repeats = 3,number = 5,search='random')
  
  cv_grid <- expand.grid(nrounds = seq(2000,3000,by=100),
                         max_depth = seq(1,10,1),
                         eta = 1,
                         gamma = seq(0.01,1,10),
                         colsample_bytree = seq(0.01,1,10),
                         min_child_weight=seq(0,10,1),
                         subsample=seq(0.5,1,10)
  )
  
    train(y ~.,
         data=train_data,
         method="xgbTree",
         metric = "RMSE",
         trControl=cv_ctrl,
         tuneLength = reps
  )
}


predict_time <- function(data){
  times <- c()
  iter <- seq(1,251,by=50)
  for(i in iter){
    print('iteration')
    times <- append(times,system.time(fit_model(data,i))['elapsed'])
  }
  plot(iter,times)
  return(times)
}

use_stack <- function(data,train_stack,test_stack){
  train_data <- data.frame(data,read.csv(train_stack))
  test_data <- data.frame(predict_these,read.csv(test_stack))
  return(list(train_data,test_data))
}

#list[data,predict_these] = use_stack(data,'restricted_stack_predictors_train.csv','restricted_stack_predictors_test.csv')
#list[train_data,test_data] = partition_data(data,0.8)

cv_ctrl <- trainControl(method = "repeatedcv", repeats = 3,number = 5,search='random')

#cv_grid <- expand.grid(nrounds = seq(2000,3000,by=100),
##                        max_depth = seq(1,10,1),
#                        eta = 1,
##                        gamma = seq(0.01,1,10),
#                        colsample_bytree = seq(0.01,1,10),
#                        min_child_weight=seq(0,10,1),
#                        subsample=seq(0.5,1,10)
#                      )

model <- train(y ~.,
                 data=data,
                 method="xgbTree",
                 metric = "RMSE",
                 trControl=cv_ctrl,
                 tuneLength = 100
                 )

turn_into_csv(predict(model,predict_these[,-1]))
