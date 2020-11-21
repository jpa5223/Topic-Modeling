#Load the required library
library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(xgboost)
library(caret)
library(dplyr)
library(e1071)

# This is the function for embedding the test data to the server. (Server provided by Dr. Francis Wham)
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