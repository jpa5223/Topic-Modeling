#Load the required library
library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(e1071)

# This is the function for embedding the test data to the server (Server provided by Dr. Francis Wham).
getEmbeddings<-function(text){
  input <- list(
    instances =list( text)
  )
  res <- POST("https://dsalpha.vmhost.psu.edu/api/use/v1/models/use:predict", body = input,encode = "json", verbose())
  emb<-unlist(content(res)$predictions)
  emb
}


#open trainingd_data and test_data from the raw folder
training_data<-fread('./project/volume/data/raw/training_data.csv')
test_data<-fread('./project/volume/data/raw/test_file.csv')

#in order to embedding, need to create new variable
emb_dt<-NULL

#you only want to send all the text for the datasets, thats why we need to append the dataset.
text_data <- rbind(training_data, test_data, fill =TRUE)

#save the id of the training and test data for later.
id <- text_data$id

#run the function to send the all the text data to the server in order to get the text embedding through neural network.
for (i in 1:length(text_data$text)){
  emb_dt<-rbind(emb_dt,getEmbeddings(text_data$text[i]))
}

#because embedding the data to the server takes more than  2 hours, it is good to save the embedded data to csv file.
fwrite(emb_dt,'./project/volume/data/interim/emb_dt.csv')
emb_dt<-fread('./project/volume/data/interim/emb_dt.csv')


# do jittering to remove redundant rows and unnecessary data. 
#because value of emb_dt is small, it is good to set the factor to 0.01
jittered_data<-data.frame(lapply(emb_dt, jitter,factor=0.01))

# check the row of the jittered data in order to see if the row of the cleaned data and the appended data have the same row
nrow(jittered_data)

# we need to do pca in order to process to t-SNE
pca<-prcomp(jittered_data, center =TRUE, scale =TRUE)

# because PCA itself is encoded we want to unclass the pca by its x values and make it to the data table.
pca_dt<-data.table(unclass(pca)$x)
pca_dt$cars<- text_data$subredditcars

#just to see how does my pca for training dataset look like
ggplot(pca_dt[0:200,],aes(x=PC1,y=PC2,col=cars))+geom_point()

#remove the subreddit car data because we do not need the data for that when running the Rtnse.
pca_dt$cars <- NULL

# we send the pca_dt to t-SNE and we turn the pca to False because we already ran pca above
# perplexity is set to 50 because optimal perplexity is from 5~50.
#However, there is a lot of data in the dataset, so I decided to set my perplexity 50 and also
#increase my max_iteration to 5000 because when runnign multiple times, I realized that default value
#of 1000 was too less to reduce the error of the tsne, but too high will be unnecessary so I believed that 
#5000 was the good amount for the hyperparameter.
tsne<-Rtsne(pca_dt, pca = F, perplexity = 50, verbose= TRUE, max_iter= 5000)

# change the tsne to data table for data processing
tsne_dt<-data.table(tsne$Y)

#see my ggplot for TSNE
#looking at the plot I can see it is well clustered because we can have clear distiction of each classes
#even though we can not see the color because we do not know the label for the test dataset.
ggplot(tsne_dt,aes(x=V1,y=V2))+geom_point()

# add saved id back to tsne_dt
tsne_dt$id<-id

# gather the dummy y values (one hot encoded) into one column for later processing
subreddits <- names(text_data[,3:12])[apply(text_data[,3:12], 1, match, x = 1)]
tsne_dt$subreddits <- subreddits


#This function is to for feature engineering, you want to analyze the text and think about
#which word might help to get the better predicted result and improve the multi class log loss
text_data$wordcars<-0
text_data$wordcars[grep("cars",text_data$text)]<-1
text_data$wordscience<-0
text_data$wordscience[grep("science",text_data$text)]<-1
text_data$wordmachinelearning<-0
text_data$wordmachinelearning[grep("machine learning",text_data$text)]<-1
text_data$wordmars<-0
text_data$wordmars[grep("Mars",text_data$text)]<-1
text_data$wordpython<-0
text_data$wordpython[grep("python",text_data$text)]<-1
text_data$wordrecipe<-0
text_data$wordrecipe[grep("recipes",text_data$text)]<-1
text_data$wordcook<-0
text_data$wordcook[grep("cook",text_data$text)]<-1
text_data$wordfood<-0
text_data$wordfood[grep("food",text_data$text)]<-1
text_data$wordmagic<-0
text_data$wordmagic[grep("magic",text_data$text)]<-1
text_data$wordgt<-0
text_data$wordgt[grep("gt",text_data$text)]<-1
text_data$wordeGPU<-0
text_data$wordeGPU[grep("eGPU",text_data$text)]<-1
text_data$wordoptimization<-0
text_data$wordoptimization[grep("optimization",text_data$text)]<-1
text_data$wordSUV<-0
text_data$wordSUV[grep("SUV",text_data$text)]<-1
text_data$wordstatistic<-0
text_data$wordstatistic[grep("statistic",text_data$text)]<-1

head(text_data)
#Write out to interim

tsne_dt$wordcars<-text_data$wordcars
tsne_dt$wordscience<-text_data$wordscience
tsne_dt$wordmachinelearning<-text_data$wordmachinelearning
tsne_dt$wordmars<-text_data$wordmars
tsne_dt$wordinsects<-text_data$wordinsects
tsne_dt$wordpython<-text_data$wordpython
tsne_dt$wordrecipe<-text_data$wordrecipe
tsne_dt$wordcook<-text_data$wordcook
tsne_dt$wordfood<-text_data$wordfood
tsne_dt$wordmagic<-text_data$wordmagic
tsne_dt$wordgt<-text_data$wordgt
tsne_dt$wordeGPU<-text_data$wordeGPU
tsne_dt$wordoptimization<-text_data$wordoptimization
tsne_dt$wordSUV<-text_data$wordSUV
tsne_dt$wordstatistic<-text_data$wordstatistic
 
# write your tsne_dt file to interim.
fwrite(sub,'./project/volume/data/interim/tsne_dt.csv')

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

