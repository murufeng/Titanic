train.data=read.csv("D:\\R\\train.csv",na.strings = c("NA",""))
str(train.data)
train.data$Survived=factor(train.data$Survived)
train.data$Pclass=factor(train.data$Pclass)
is.na(train.data$Age)
sum(is.na(train.data$Age)==TRUE)
sum(is.na(train.data$Age)==TRUE)/length(train.data$Age)

sapply(train.data, function(df){
  sum(is.na(df)==TRUE)/length(df)
})

install.packages("Amelia")
library(Amelia)

missmap(train.data,min="Missing Map")

AmeliaView()

table(train.data$Embarked,useNA = "always")
train.data$Embarked[which(is.na(train.data$Embarked))]='S';
table(train.data$Embarked,useNA = "always")

train.data$Name=as.character(train.data$Name)
table_words=table(unlist(strsplit(train.data$Name,'\\s+')))
sort(table_words[grep('\\.',names(table_words))],decreasing = TRUE)

library(stringr)
tb=cbind(train.data$Age,str_match(train.data$Name,"[a-zA-Z]+\\."))
table(tb[is.na(tb[,1]),2])

mean.mr=mean(train.data$Age[grepl("Mr\\.",train.data$Name)&!is.na(train.data$Age)])
mean.mrs=mean(train.data$Age[grepl("Mrs\\.",train.data$Name)&!is.na(train.data$Age)])
mean.dr=mean(train.data$Age[grepl("Dr\\.",train.data$Name)&!is.na(train.data$Age)])
mean.miss=mean(train.data$Age[grepl("Miss\\.",train.data$Name)&!is.na(train.data$Age)])
mean.master=mean(train.data$Age[grepl("Master\\.",train.data$Name)&!is.na(train.data$Age)])

train.data$Age[grepl("Mr\\.",train.data$Name)&is.na(train.data$Age)]=mean.mr
train.data$Age[grepl("Mrs\\.",train.data$Name)&is.na(train.data$Age)]=mean.mrs
train.data$Age[grepl("Dr\\.",train.data$Name)&is.na(train.data$Age)]=mean.dr
train.data$Age[grepl("Miss\\.",train.data$Name)&is.na(train.data$Age)]=mean.miss
train.data$Age[grepl("Master\\.",train.data$Name)&is.na(train.data$Age)]=mean.master


barplot(table(train.data$Survived),main="Passenger Survival",names=c("Perished","Survived"))

barplot(table(train.data$Pclass),main="Passenger Class",names=c("first","second","third"))

barplot(table(train.data$Sex),main="PassengerGender")

hist(train.data$Age,main="Passenger Age",xlab="Age")

barplot(table(train.data$SibSp),main="Passenger Siblings")

barplot(table(train.data$Parch),main="Passenger Parch")

hist(train.data$Fare,main="Passenger Fare",xlab="Fare")

barplot(table(train.data$Embarked),main="Port of Embarkation")

counts=table(train.data$Survived,train.data$Sex)
barplot(counts,col=c("darkblue","red"),legend=c("Perished","Survived"),main = "Passenger Survival by Sex")

counts=table(train.data$Survived,train.data$Pclass)
barplot(counts,col=c("darkblue","red"),legend=c("Perished","Survived"),main = "Titanic Class Bar Plot")

counts=table(train.data$Sex,train.data$Pclass)
barplot(counts,col=c("darkblue","red"),legend=rownames(counts),main = "Passenger Gender by Class")

hist(train.data$Age[which(train.data$Survived=="0")],main = "Passenger Age Histogram",xlab = "Age",ylab = "Count",col="blue",breaks = seq(0,80,by=2))

hist(train.data$Age[which(train.data$Survived=="1")],col="blue",add=T,breaks = seq(0,80,by=2))

boxplot(train.data$Age~train.data$Survived,main="Passenger Survival by Age",xlab="Survived",ylab="Age")

train.child=train.data$Survived[train.data$Age<13]
length(train.child[which(train.child==1)])/length(train.child)

train.youth=train.data$Survived[train.data$Age>=15&train.data$Age<25]
length(train.youth[which(train.youth==1)])/length(train.youth)

train.adult=train.data$Survived[train.data$Age>=20&train.data$Age<65]
length(train.adult[which(train.adult==1)])/length(train.adult)

train.senior=train.data$Survived[train.data$Age>=65]
length(train.senior[which(train.senior==1)])/length(train.senior)

mosaicplot(train.data$Pclass~train.data$Survived,main = "Passenger Survival Class",color=TRUE,xlab="Pclass",ylab="Survived")

c("基于决策树预测获救乘客")
split.data=function(data,p=0.7,s=666){
  set.seed(s)
  index=sample(1:dim(data)[1])
  train=data[index[1:floor(dim(data)[1]*p)],]
  test=data[index[((ceiling(dim(data)[1]*p))+1):dim(data)[1]],]
  return(list(train=train,test=test))
}

allset=split.data(train.data,p=0.7)
trainset=allset$train
testset=allset$test

install.packages('party')
require('party')

 library(grid)
 library(mvtnorm)
library(modeltools)
 library(stats4)
library(strucchange)
library(zoo)
library(party)

train.ctree=ctree(Survived~Pclass+Sex+Age+SibSp+Fare+Parch+Embarked,data=trainset)
train.ctree

plot(train.ctree,main="Conditional inference tree of Titanic Dataset")

c("基于混淆矩阵验证预测结果的准确性")
ctree.predict=predict(train.ctree,testset)
library(lattice)
library(ggplot2)
library('caret')
confusionMatrix(ctree.predict,testset$Survived)

c("3.使用ROC曲线评估性能")
train.ctree.pred=predict(train.ctree,testset)
train.ctree.prob=1-unlist(treeresponse(train.ctree,testset),use.names = F)[seq(1,nrow(testset)*2,2)]

install.packages('ROCR')
library(gplots)
library(ROCR)
train.ctree.prob.rocr=prediction(train.ctree.prob,testset$Survived)
train.ctree.pref=performance(train.ctree.prob.rocr,"tpr","fpr")
train.ctree.auc.perf=performance(train.ctree.prob.rocr,measure="auc",x.measure="cutoff")

plot(train.ctree.pref,col=2,colorize=T,main=paste("AUC:",train.ctree.auc.perf@y.values))










               
               
