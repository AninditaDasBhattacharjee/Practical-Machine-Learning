# Practical-Machine-Learning
Prediction Assignment Writeup

Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

Data 
The training data for this project are available here: 
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Analysis
First, we begin by downloading the training and testing datasets and upload them in the R Studio.
1. Then, we load some useful packages using:
> # Require the necessary packages
> require(data.table)
Loading required package: data.table
data.table 1.12.8 using 16 threads (see ?getDTthreads).  Latest news: r-datatable.com
> require(dplyr)
Loading required package: dplyr

Attaching package: ‘dplyr’

The following objects are masked from ‘package:data.table’:

    between, first, last

The following objects are masked from ‘package:stats’:

    filter, lag

The following objects are masked from ‘package:base’:

    intersect, setdiff, setequal, union

> require(caret)
Loading required package: caret
Loading required package: ggplot2
Loading required package: lattice
2. Then we will load the data into memory.
> # Load the training data from working directory
> train <- read.csv("training.csv")
> 
> # Load the testing data from working directory
> test <- read.csv("testing.csv")
3. Next, we need to prepare the data for modelling. We can see is a lot of data with NA or empty values. Let’s remove those.
> Percentage_max_NA = 90
> maxNACount <- nrow(train) / 100 * Percentage_max_NA
> removeColumns <- which(colSums(is.na(train) | train=="") > maxNACount)
> training.cleaned <- train[,-removeColumns]
> testing.cleaned <- test[,-removeColumns]

> #that reduces the columns to only 60 columns
> dim(training.cleaned)
[1] 19622    60

> dim(testing.cleaned)
[1] 20 60

> #Investigating the data we can see that the seven first columns have a sequencial number (the first)
> #and variations of the timestamp that we are not using for this analysis so we will eliminate those columns remaining 53
> trainOK<-training.cleaned[,-c(1:6)]
> testOK<-testing.cleaned[,-c(1:6)]
> dim(trainOK);dim(testOK)
[1] 19622    54
[1] 20 54

> exerCorrmatrix<-cor(trainOK[sapply(trainOK, is.numeric)])  
> corrplot(exerCorrmatrix,order="original", method="circle", type="lower", tl.cex=0.45, tl.col="black", number.cex=0.25) 

 
4. Now we will split the current training in a test and train set in order to validate it.
> set.seed(2022)
> inTrain<-createDataPartition(trainOK$classe, p=3/4, list=FALSE)
> train<-trainOK[inTrain,]
> valid<-trainOK[-inTrain,]
After analysing the principal components, we got that 25 components are necessary to capture 0.95 of the variances. But it demands a lot of machine processing so, we decided by a 0.80 thresh to capture 80% of the variance using 13 components.
> PropPCA<-preProcess(train[,-54],method="pca", thresh=0.8)
> PropPCA
Created from 14718 samples and 53 variables

Pre-processing:
  - centered (53)
  - ignored (0)
  - principal component signal extraction (53)
  - scaled (53)

PCA needed 13 components to capture 80 percent of the variance
Pre-processing

> #create the preProc object, excluding the response (classe)
> preProc  <- preProcess(train[,-54], 
+                        method = "pca",
+                        pcaComp = 13, thresh=0.8) 
> #Apply the processing to the train and test data, and add the response 
> #to the dataframes
> train_pca <- predict(preProc, train[,-54])
> train_pca$classe <- train$classe
> #train_pca has only 13 principal components plus classe
> valid_pca <- predict(preProc, valid[,-54])
> valid_pca$classe <- valid$classe
> #valid_pca has only 13 principal components plus classe

Model examination

At this point, we have clean data that we can use for building models and we will use the Random Forest model
> #create Random Forest model
> start <- proc.time()
> fitControl<-trainControl(method="cv", number=5, allowParallel=TRUE)
> fit_rf<-train(classe ~., data=train_pca, method="rf", trControl=fitControl)
> print(fit_rf, digits=4)
Random Forest 

14718 samples
   13 predictor
    5 classes: 'A', 'B', 'C', 'D', 'E' 

No pre-processing
Resampling: Cross-Validated (5 fold) 
Summary of sample sizes: 11774, 11775, 11774, 11774, 11775 
Resampling results across tuning parameters:

  mtry  Accuracy  Kappa 
   2    0.9596    0.9490
   7    0.9534    0.9411
  13    0.9453    0.9308

Accuracy was used to select the optimal model using the largest value.
The final value used for the model was mtry = 2.

> proc.time() - start
    user   system  elapsed 
 169.718    8.189 2753.557 

> predict_rf<-predict(fit_rf,valid_pca)  
> (conf_rf<-confusionMatrix(as.factor(valid_pca$classe), predict_rf))
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1369   10    8    7    1
         B   15  917   14    3    0
         C   13   20  810    7    5
         D    0    0   41  762    1
         E    0    5   13    5  878

Overall Statistics
                                          
               Accuracy : 0.9657          
                 95% CI : (0.9603, 0.9707)
    No Information Rate : 0.2849          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9567          
                                          
 Mcnemar's Test P-Value : 3.222e-07       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9800   0.9632   0.9142   0.9719   0.9921
Specificity            0.9926   0.9919   0.9888   0.9898   0.9943
Pos Pred Value         0.9814   0.9663   0.9474   0.9478   0.9745
Neg Pred Value         0.9920   0.9912   0.9812   0.9946   0.9983
Prevalence             0.2849   0.1941   0.1807   0.1599   0.1805
Detection Rate         0.2792   0.1870   0.1652   0.1554   0.1790
Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
Balanced Accuracy      0.9863   0.9776   0.9515   0.9809   0.9932

> (accuracy_rf<-conf_rf$overall['Accuracy'])
 Accuracy 
0.9657423

We can now say that for this data-set, random forest method has an accuracy of 0.96.
Prediction on Testing Set
Applying the Random Forest to predict the outcome variable classe for the test set to make predictions on the 20 data points from the original testing dataset.

> test_pca <- predict(preProc, testOK[,-54])
> test_pca$problem_id <- testOK$problem_id
> (predict(fit_rf, test_pca))
 [1] B A C A A E D B A A B C B A E E A B B B
Levels: A B C D E

