# Proposal

###  Website link to download dataset
## https://archive.ics.uci.edu/ml/datasets/Las+Vegas+Strip

##

library(ggplot2)
library(nnet)
library("e1071")

## *Adjust to the folder were your dataset is at (if in desktop or downloads)* 
setwd("~/Desktop/Projects/School/Data Mining")

tripdf= read.table("LasVegasTripAdvisorReviews-Dataset (1).csv", sep = ";", header = TRUE)

class(tripdf)

## Data exploration

summary(tripdf)
str(tripdf)
dim(tripdf)
head(tripdf,5)
sum(tripdf$Score)


## Tables used on website
numericalSUMTRIP <- summary(tripdf[,c(2,3,4,5, 16, 18)])



catSUM <- summary(tripdf[,c(1,6,7,8,9,10,11,12,13,14,15,17,19,20)])
catSUM2 <- summary(tripdf[,c(12,13,14,15,17,19,20)])
table(tripdf$User.country)
table(tripdf$Nr..reviews)


## Box plots

Nr_reviews_box <- boxplot(tripdf[,"Nr..reviews"], ylab="Count", main= "User Reviews Written", col = 'powderblue')
Nr_hotel_reviews_box <- boxplot(tripdf[,"Nr..hotel.reviews"], ylab="Count", main= "Total Hotel Reviews", col =  'powderblue')
Helpful_votes_box <- boxplot(tripdf[,"Helpful.votes"], ylab="Count", main="Helpful votes from User", col= 'powderblue')
Nr_rooms_box <- boxplot(tripdf[, "Nr..rooms"], ylab="Count", main= "Hotels Number of Rooms", col = 'powderblue')
Member_years_box <- boxplot(tripdf[,"Member.years"], ylab="Count", main= "Member number of years", col = 'powderblue')
Score_box <- boxplot(tripdf[,"Score"],ylab="Count", main= "Score", col = 'powderblue')

## Histograms

Nr_reviews_hist <- hist(tripdf[,"Nr..reviews"], main = "Histogram of User Reviews", xlab= "User Reviews Written", col = 'powderblue')
Nr_hotel_reviews_hist <- hist(tripdf[,"Nr..hotel.reviews"],main = "Histogram of Total Hotel Reviews", xlab= "Total Hotel Reviews", col = 'powderblue')
Helpful_votes_hist <- hist(tripdf[,"Helpful.votes"], main = "Histogram of Helpful Votes", xlab="Helpful votes from User", col = 'powderblue')
Nr_rooms_hist <- hist(tripdf[, "Nr..rooms"], main = "Histogram of Number of Rooms per Hotel", xlab= "Hotels Number of Rooms", col = 'powderblue')
Member_years_hist <- hist(tripdf[,"Member.years"],  main = "Histogram of Years of member", xlab= "Member number of years", col = 'powderblue')
Score_hist <- hist(tripdf[,"Score"],main = "Histogram of Score", xlab= "Score", col = 'powderblue')



## Categorical variable exploration

catVars<- c(1,6,7,8,9,10,11,12,13,14,15,17,19,20)
plot(tripdf[,1])
plot(tripdf[,6])
plot(tripdf[,7])
plot(tripdf[,8])
plot(tripdf[,9])
plot(tripdf[,10])
plot(tripdf[,11])
plot(tripdf[,12])
plot(tripdf[,13])
plot(tripdf[,14])
plot(tripdf[,15])
plot(tripdf[,17])
plot(tripdf[,19])
plot(tripdf[,20])


hist(tripdf$Member.years, col = "powderblue")

## fix negative number found in data 
## looks like its a missing value so I'll replace with zero

tripdf[76, "Member.years"] <-0

## multinomial logistic regression

tripdf$Score<- factor(tripdf$Score)
tripdf$out <- relevel( tripdf$Score, ref = "1")



## mulitnomial w/ all variables
full_MNLmodel <- multinom(Score ~ . , data = tripdf)

## multinomial w/ sum variables
dscrt_MNLmodel <- multinom(out~Period.of.stay+Traveler.type+Pool+Gym+Tennis.court+
                      Spa+Casino+Free.internet+Nr..rooms+Review.month+
                      Review.weekday, data = tripdf)

summary(full_MNLmodel)
summary(dscrt_MNLmodel)
## lower AIC And Deviance for full model

## predict full model scores
findB<- predict(full_MNLmodel,tripdf)
findBp<- predict(full_MNLmodel,tripdf, type = "prob")
summary(findB)

## predict discrete model scores
findC<- predict(dscrt_MNLmodel, tripdf)
findCp<- predict(dscrt_MNLmodel,tripdf, type = "prob")
summary(findC)

## predicted probs for full model  
predict(full_MNLmodel, tripdf, "probs")
exp(coef(full_MNLmodel));exp(confint(full_MNLmodel))

## predicted probs for discrete model 
trip_c<- tripdf[,c("Period.of.stay","Traveler.type","Pool","Gym","Tennis.court","Spa","Casino",
                      "Free.internet","Nr..rooms","Review.month","Review.weekday")]
predict(dscrt_MNLmodel,trip_c, "probs")
exp(coef(dscrt_MNLmodel));exp(confint(dscrt_MNLmodel))

str(tripdf)

## predict full model scores
findB<- predict(full_MNLmodel,tripdf)
findBp<- predict(full_MNLmodel,tripdf, type = "prob")
summary(findB)

## predict discrete model scores
findC<- predict(dscrt_MNLmodel, tripdf)
findCp<- predict(dscrt_MNLmodel,tripdf, type = "prob")
summary(findC)


## observed(given) Score 
summary(tripdf$Score)

## misclassification rate full model
cmb<-table(predict(full_MNLmodel),tripdf$Score)
1-sum(diag(cmb))/sum(cmb)

## misclassification rate discrete model
cmC<-table(predict(dscrt_MNLmodel),tripdf$Score)
1-sum(diag(cmC))/sum(cmC)


## Full Model results
zB<- summary(full_MNLmodel)$coefficients/summary(full_MNLmodel)$standard.errors
pB<- (1- pnorm(abs(zB),0,1))*2
pB
exp(coef(full_MNLmodel))


## Discrete model results
z<- summary(dscrt_MNLmodel)$coefficients/summary(dscrt_MNLmodel)$standard.errors
p<- (1- pnorm(abs(z),0,1))*2
p
exp(coef(dscrt_MNLmodel))

## Support Vector Machine


subset <- sample(nrow(tripdf), nrow(tripdf)* .8)
col<- c("Member.years", "Helpful.votes","Score")
tripdf_train <- tripdf[subset, col]
tripdf_test <- tripdf[-subset,]


svmfit <-svm(Score~., data=tripdf , probability= TRUE, cost=.001)
summary(svmfit)

plot(svmfit, data = tripdf,Helpful.votes~Member.years )


qplot(tripdf$Member.years,tripdf$Helpful.votes, col=tripdf$Score)

cols<-c("Member.years", "Traveler.type", "Score")

plot(svmfit,tripdf[,c("Pool","Traveler.type")])


svmfit1 <-svm(Score~., data=tripdf_train, probability= TRUE)

x <- subset(tripdf, select = -Score)
y <- tripdf$Score
model <- svm(tripdf$Period.of.stay, tripdf$Score, probability = TRUE)

# pred svm
predSVM <- predict(svmfit,tripdf, type="class")
tab<- table(Predicted= predSVM, Actual = tripdf$Score)
1-sum(diag(tab))/sum(tab)


#tune
tuned<- tune(svm,Score~.,data = tripdf_train,ranges = list(cost=c(.001,.01,.1,1,10,100)))
summary(tuned)

