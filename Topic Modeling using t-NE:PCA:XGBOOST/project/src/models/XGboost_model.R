#Load the libraries
library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(e1071)

#divide the merged data table back to train and test
train <- tsne_dt[0:200,]
test <- tsne_dt[201:20755,]

#set the label for the y value for later processing the XGBoost
outcome <- train$subreddits
reddits<-as.numeric(as.factor(outcome))-1

#make unnecessary data null
train$id <- NULL
test$id<- NULL
train$subreddits <-NULL
test$subreddits<-NULL

# change the data table to metrics for processing the DMatrix for the XGboost
train.matrix = as.matrix(train)
mode(train.matrix) = "numeric"
test.matrix = as.matrix(test)
mode(test.matrix) = "numeric"



# make Dmatrix for the XGBoost with the train and test Matrix.
dtrain <- xgb.DMatrix(train.matrix,label=reddits,missing=NA)
dtest <- xgb.DMatrix(test.matrix,missing=NA)
ytrain <- xgb.DMatrix(ytrain.matrix,missing=NA)

#classes are needed when we want to use multiclass loss in XGboost because we want to know how many y value
#we want to produce
classes<-factor(unique(reddits))

# The parameter for the XGBoost
param <- list(objective = "multi:softprob", #obejective of my model.
              booster = 'gbtree',
              eval_metric = "mlogloss", #evaluation metrics I will use to check my error.
              num_class = length(classes),  # number of class needed for analysis.
              eta = 0.02,
              max_depth = 20,
              colsample_bytree = 1,
              subsample = 0.7
)

#This model is to test my training set to optimize my hyper parameters
XGBcv<-xgb.cv( params=param,nfold=300,nrounds=1000, missing=NA,data=dtrain, print_every_n=1, 
               verbose =  TRUE, prediction = TRUE)

#test my model
prediction <- data.frame(XGBcv$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = reddits + 1)
head(prediction)

confusionMatrix(factor(prediction$max_prob),
                factor(prediction$label),
                mode = "everything")

#Make watchlist to observe the training set so we are able to observe the error.
watchlist <- list(train = dtrain)

#when doing the traing, after playing with the hyperparameters around, I realized 
#it is good to optimize my n_rounds to 100 because too hight value for round will cause
#over_fitting and too low round will cause under-fitting.

XGBm<-xgb.train(params=param, nrounds=300,missing=NA,data=dtrain, label = reddits,watchlist=watchlist,
                print_every_n=1,early_stopping_rounds = 20)

# run real prediction with dtest
pred<-predict(XGBm, newdata = dtest)

#make the pred to data table but because it is given in one column, we want to split it to 10 classe
probs<- as.data.frame(t(matrix(pred,ncol=length(pred)/10)))

#add back the id to see the samples
probs$id <- test_data$id
probs
#read the submission file from the given dataset for preparing submission file.
sub<-fread('./project/volume/data/raw/example_sub.csv')

# because it is done with label encoding, the data is in order. However, for comparison
# I read about 100 texts and compared them and saw good result.
# after comparison, I assigned them into right place for the submission file.
sub$subredditcars <- probs$V1
sub$subredditCooking <- probs$V2
sub$subredditMachineLearning <-probs$V3
sub$subredditmagicTCG <-probs$V4
sub$subredditpolitics <-probs$V5
sub$subredditReal_Estate <- probs$V6
sub$subredditscience <- probs$V7
sub$subredditStockMarket <- probs$V8
sub$subreddittravel<- probs$V9
sub$subredditvideogames<- probs$V10

#Finally, write out the submission and it is over.
fwrite(sub,'./project/volume/data/processed/result.csv')

