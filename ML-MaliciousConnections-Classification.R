# clean environment
rm(list = ls())

##      ENVIRONMENT      ##
#_______________________________________________________________________________
# get & set Directory
getwd()
setwd()

# install required packages
# install.packages(c("tidyverse","caret","glmnet","rpart","rpart.plot"))
# install.packages(c("ipred","ranger"))

# load Libraries
library(tidyverse)
library(caret)
library(glmnet)
library(rpart) 
library(rpart.plot)  
library(ipred)
library(ranger)

# import data
myDataRaw <- read.csv("MLData2023.csv", stringsAsFactors = TRUE)

# Relocate all categorical(factor) features to the left
myDataRaw <- myDataRaw %>%
    select(where(is.factor), 
           everything()); myDataRaw

# Check dimensions and structure of data set
dim(myDataRaw) 
str(myDataRaw)

##      CLEAN DATASET      ##
#_______________________________________________________________________________

# remove IPV6.Traffic variable from data set
myData <- subset(myDataRaw, select = -IPV6.Traffic)

# mask invalid values in Assembled.Payload.Size as "NA"
str(myData$Assembled.Payload.Size)
invAPS <-  which(myData$Assembled.Payload.Size <= 0)
myData$Assembled.Payload.Size <- replace(myData$Assembled.Payload.Size, invAPS, NA)

# mask invalid values in Operating.System as "NA"
str(myData$Operating.System)
levels(myData$Operating.System)
invOS <-  which(myData$Operating.System == "-")
myData$Operating.System <- replace(myData$Operating.System, invOS, NA)

# filter out values with invalid class
myData <- filter(myData, Class == 0 | Class == 1)
str(myData)

# merge categories on Operating.System feature
levels(myData$Operating.System)
myData$Operating.System <- fct_collapse(myData$Operating.System,
                                        Windows = c("Windows (Unknown)", 
                                                    "Windows 10+",
                                                    "Windows 7"),
                                        Others = c("iOS", 
                                                   "Linux (unknown)", 
                                                   "Other"))

# merge categories on Connection.State feature
levels(myData$Connection.State)
myData$Connection.State <- fct_collapse(myData$Connection.State, 
                                        Others = c("INVALID",
                                                   "NEW",
                                                   "RELATED"))

# filter out observations containing NA values
MLData2023_cleaned <- na.omit(myData)
str(MLData2023_cleaned)
summary(MLData2023_cleaned)

#drop unused levels on Operating.System feature
MLData2023_cleaned$Operating.System <- forcats::fct_drop(MLData2023_cleaned$Operating.System)


##      DEFINE TEST & TRAINING SAMPLES      ##
#_______________________________________________________________________________

# Separate non-malicious and malicious observations
dat.class0 <- MLData2023_cleaned %>% filter(Class == 0) # non-malicious 
dat.class1 <- MLData2023_cleaned %>% filter(Class == 1) # malicious

##       UNBALANCED TRAINING SET     ##
#_______________________________________________________________________________
set.seed(10564447)

# 20000 unbalanced training sample, 19800 non-malicious and 200 malicious 
# randomly select rows of 19800 negative observations
rows.train0 <- sample(1:nrow(dat.class0), 
                      size = 19800, 
                      replace = FALSE) 

# randomly select rows of 200 positive observations
rows.train1 <- sample(1:nrow(dat.class1), 
                      size = 200, 
                      replace = FALSE)

# define sets according to sample rows
train.class0 <- dat.class0[rows.train0,] # Non-malicious samples 
train.class1 <- dat.class1[rows.train1,] # Malicious samples 

# combine positive and negative samples
mydata.ub.train <- rbind(train.class0, train.class1)
# factor and rename levels of Class variable
mydata.ub.train <- mydata.ub.train %>% 
    mutate(Class = factor(Class, labels = c("NonMal","Mal")))

# write unbalanced training data to csv file 
write.csv(mydata.ub.train,"mydata.ub.train.csv")


##       BALANCED TRAINING SET     ##
#_______________________________________________________________________________
set.seed(77)

# 39600 balanced training samples, 19800 non-malicious and malicious samples each

# Bootstrapping the class 1 observations in the training data
# The same row can be selected multiple times, increasing the number of class 1 observations.
train.class1_2 <- train.class1[sample(1:nrow(train.class1), 
                                      size = 19800,
                                      replace = TRUE),]

# combine positive and negative samples
mydata.b.train <- rbind(train.class0, 
                        train.class1_2)
# factor and rename levels of Class variable
mydata.b.train <- mydata.b.train %>% 
    mutate(Class = factor(Class, labels = c("NonMal","Mal")))

# write balanced training data to csv file 
write.csv(mydata.b.train,"mydata.b.train.csv")

##       TEST SET     ##
#_______________________________________________________________________________

# exclude rows used in training set
test.class0 <- dat.class0[-rows.train0,] 
test.class1 <- dat.class1[-rows.train1,]

# combine positive and negative test samples
mydata.test <- rbind(test.class0, 
                     test.class1)
# factor and rename levels of Class variable
mydata.test <- mydata.test %>% 
    mutate(Class = factor(Class, labels = c("NonMal","Mal")))

# write test data to csv file 
write.csv(mydata.test,"mydata.test.csv")

##      RANDOM FOREST UNBALANCED      ##
#_______________________________________________________________________________
set.seed(77)

# Create a search grid for the tuning parameters
grid.RF.unB <- expand.grid(num.trees = c(200,350,500),  #Number of trees
                          mtry = c(2,6,12),  #Split rule
                          min.node.size = seq(2,10,2),  #Tree complexity
                          replace = c(TRUE, FALSE),  #Sampling with or without replacement
                          sample.fraction = c(0.5,0.75,1),  #Sampling fraction
                          OOB.misclass = NA,   #Column to store the OOB RMSE
                          test.sens = NA,  #Column to store the test Sensitivity
                          test.spec = NA,  #Column to store the test Specificity
                          test.acc = NA) #Column to store the test Accuracy

#Check the dimension
dim(grid.RF.unB)  
#View the search grid
View(grid.RF.unB)  

set.seed(77)
for (I in 1:nrow(grid.RF.unB))
{
    RF.unB <- ranger(Class~., 
                    data = mydata.ub.train, 
                    num.trees = grid.RF.unB$num.trees[I],
                    mtry = grid.RF.unB$mtry[I],
                    min.node.size = grid.RF.unB$min.node.size[I],
                    replace = grid.RF.unB$replace[I],
                    sample.fraction = grid.RF.unB$sample.fraction[I],
                    seed = 77,
                    respect.unordered.factors = "order")
    
    grid.RF.unB$OOB.misclass[I] <- RF.unB$prediction.error %>% round(5)*100
    
    #Test classification
    test.RF.unB <- predict(RF.unB, 
                           data = mydata.test)$predictions; #Predicted classes
    
    #Summary of confusion matrix
    cf.RF.unB <- confusionMatrix(test.RF.unB %>% relevel(ref="Mal"),
                                 mydata.test$Class %>% relevel(ref="Mal")); 
    
    prop.cf.RF.unB <- cf.RF.unB$table %>% prop.table(2)
    grid.RF.unB$test.sens[I] <- prop.cf.RF.unB[1,1] %>% round(5)*100  #Sensitivity
    grid.RF.unB$test.spec[I] <- prop.cf.RF.unB[2,2] %>% round(5)*100  #Specificity
    grid.RF.unB$test.acc[I] <- cf.RF.unB$overall[1] %>% round(5)*100  #Accuracy
}

# Sort the results by the OOB misclassification error and view the top 10 results
grid.RF.unB[order(grid.RF.unB$OOB.misclass,decreasing=FALSE)[1:10],] 

set.seed(77)
# train model with top performing parameters
RF.unB.TOP <- ranger(Class~., 
                     data = mydata.ub.train, 
                     num.trees = 200, 
                     mtry = 6, 
                     min.node.size = 4, 
                     replace = FALSE, 
                     sample.fraction = 0.75, 
                     seed = 77,
                     respect.unordered.factors = "order")

# get OOB misclassification rate
OOB.misclass.unB.TOP <- RF.unB.TOP$prediction.error %>% round(5)*100

# Make predictions on the test set
test.RF.unB.TOP <- predict(RF.unB.TOP, data = mydata.test)$predictions

# Compute the confusion matrix for the test set predictions
cf.RF.unB.TOP <- confusionMatrix(test.RF.unB.TOP %>% relevel(ref="Mal"),
                                 mydata.test$Class %>% relevel(ref="Mal")); cf.RF.unB.TOP

##      RANDOM FOREST BALANCED      ##
#_______________________________________________________________________________
set.seed(77)

# Create a search grid for the tuning parameters
grid.RF.Ba <- expand.grid(num.trees = c(200,350,500),  # Number of trees
                          mtry = c(2,6,12),  # Split rule
                          min.node.size = seq(2,10,2),  # Tree complexity
                          replace = c(TRUE, FALSE),  # Sampling with or without replacement
                          sample.fraction = c(0.5,0.75,1),  # Sampling fraction
                          OOB.misclass = NA,   # Column to store the OOB misclassification rate
                          test.sens = NA,  # Column to store the test Sensitivity
                          test.spec = NA,  # Column to store the test Specificity
                          test.acc = NA) # Column to store the test Accuracy

# Check the dimension
dim(grid.RF.Ba) 
# View the search grid
View(grid.RF.Ba)  

# Run the grid search
set.seed(77)
for (I in 1:nrow(grid.RF.Ba))
{
    RF.Ba <- ranger(Class~., 
                    data = mydata.b.train, 
                    num.trees = grid.RF.Ba$num.trees[I],
                    mtry = grid.RF.Ba$mtry[I],
                    min.node.size = grid.RF.Ba$min.node.size[I],
                    replace = grid.RF.Ba$replace[I],
                    sample.fraction = grid.RF.Ba$sample.fraction[I],
                    seed = 77,
                    respect.unordered.factors = "order")
    
    grid.RF.Ba$OOB.misclass[I] <- RF.Ba$prediction.error %>% round(5)*100
    
    # Test classification
    test.RF.Ba <- predict(RF.Ba, data = mydata.test)$predictions
    
    # Summary of confusion matrix
    cf.RF.Ba <- confusionMatrix(test.RF.Ba %>% relevel(ref="Mal"),
                                mydata.test$Class %>% relevel(ref="Mal"))
    
    prop.cf.RF.Ba <- cf.RF.Ba$table %>% prop.table(2)
    grid.RF.Ba$test.sens[I] <- prop.cf.RF.Ba[1,1] %>% round(5)*100  # Sensitivity
    grid.RF.Ba$test.spec[I] <- prop.cf.RF.Ba[2,2] %>% round(5)*100  # Specificity
    grid.RF.Ba$test.acc[I] <- cf.RF.Ba$overall[1] %>% round(5)*100  # Accuracy
}

# Sort the results by the OOB misclassification error and view the top 10 results
grid.RF.Ba[order(grid.RF.Ba$OOB.misclass,decreasing=FALSE)[1:10],]

# train model with top performing parameters
set.seed(77)
RF.Ba.TOP <- ranger(Class~., 
                    data = mydata.b.train, 
                    num.trees = 200, 
                    mtry = 6, 
                    min.node.size = 6, 
                    replace = TRUE, 
                    sample.fraction = 0.5, 
                    seed = 77,
                    respect.unordered.factors = "order")

# get OOB misclassification rate
OOB.misclass.Ba.TOP <- RF.Ba.TOP$prediction.error %>% round(5)*100

# Make predictions on the test set
test.RF.Ba.TOP <- predict(RF.Ba.TOP, data = mydata.test)$predictions

# Compute the confusion matrix for the test set predictions
cf.RF.Ba.TOP <- confusionMatrix(test.RF.Ba.TOP %>% relevel(ref="Mal"),
                                mydata.test$Class %>% relevel(ref="Mal")); cf.RF.Ba.TOP


##      ELASTIC-NET UNBALANCED      ##
#_______________________________________________________________________________
set.seed(77)

# define search ranges for lambda values
lambdas <- 10^seq(-3,3,length=100)

# define search ranges for alpha values
alphas <- seq(0.1, 0.9, by = 0.1)

# define control parameters for cross validation 
cont.par.elaN.unb  <- trainControl(method = "repeatedcv", number = 5, repeats = 3)

# train the elastic net model on the unbalanced set 
elaN.unb <- train(Class~., 
                            data = mydata.ub.train,
                            method = "glmnet",
                            preProcess = NULL,
                            trControl = cont.par.elaN.unb,
                            tuneGrid = expand.grid(alpha = alphas,
                                                   lambda = lambdas))

# optimal lambda value
elaN.unb$bestTune

# plot cross validation parameter results
plot(elaN.unb, 
     xlim = c(0, 0.005), 
     ylim = c(0.997, 0.9999))

#model coefficients
coef(elaN.unb$finalModel, elaN.unb$bestTune$lambda)

# predicted probability on test data
pred.elaN.unb <- predict(elaN.unb,
                         new = mydata.test)

#confusion matrix reordering yes and no responses
cf.elaN.unb <- table(pred.elaN.unb %>% as.factor %>% relevel(ref="Mal"), 
                    mydata.test$Class %>% as.factor %>% relevel(ref="Mal")); cf.elaN.unb

#summary of confusion matrix
confusionMatrix(cf.elaN.unb)

##      ELASTIC-NET BALANCED      ##
#_______________________________________________________________________________
set.seed(77)

# define search ranges for lambda values
lambdas <- 10^seq(-3,3,length=100)

# define search ranges for alpha values
alphas <- seq(0.1, 0.9, by = 0.1)

# define control parameters for cross validation 
cont.par.elaN.Ba  <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# train the elastic net model on the balanced set 
elaN.Ba <- train(Class~., 
                 data = mydata.b.train,
                 method = "glmnet",
                 preProcess = NULL,
                 trControl = cont.par.elaN.Ba,
                 tuneGrid = expand.grid(alpha = alphas,
                                        lambda = lambdas))

# optimal lambda value
elaN.Ba$bestTune

# plot cross validation parameter results
plot(elaN.Ba, 
     xlim = c(0, 0.003), 
     ylim = c(0.990, 0.9950))

#model coefficients
coef(elaN.Ba$finalModel, elaN.Ba$bestTune$lambda)

# predicted probability on test data
pred.elaN.Ba <- predict(elaN.Ba,
                        new = mydata.test)

#confusion matrix reordering yes and no responses
cf.elaN.Ba <- table(pred.elaN.Ba %>% as.factor %>% relevel(ref="Mal"), 
                    mydata.test$Class %>% as.factor %>% relevel(ref="Mal")); cf.elaN.Ba

#summary of confusion matrix
confusionMatrix(cf.elaN.Ba)



