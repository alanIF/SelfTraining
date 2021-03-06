#comandos obrigat�rios no c�digo

#diret�rio local para salvar as bases e resultados
setwd("C:\\local_R")

#instala��o dos pacotes 

#pacote que inclui: data splitting, pre-processing, feature selection, model tuning using resampling, variable importance estimation
install.packages("caret")

#carregar os pacotes
library("caret")

#baixando o banco de dados
#para base que j� est� no R
data(iris)
dataset <- iris

#para base do UCI (formato .CSV)

#1) define the filename
#filename <- "iris.csv"

#2) load the CSV file from the local directory
#dataset <- read.csv(filename, header=FALSE)

#3) set the column names in the dataset
#colnames(dataset) <- c("Sepal.Length","Sepal.Width","Petal.Length","Petal.Width","Species")

#criando o conjunto de valida��o
# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]
#agora temos o conjunto de treinamento na vari�vel dataset e o de valida��o na vairavel validation

#o comando abaixo indica a dimens�o da base de dados, mostra primeiro a qtde de inst�ncias e depois a qtde de atributos
dim(dataset)
#o comando abaixo indica o tipo de cada atributo
sapply(dataset, class)
#comando para ver os 6 primeiros e os 6 �ltimos exemplos do conjunto de treinamento
head(dataset)
tail(dataset)
#comando para mostrar as classes da base de dados
levels(dataset$Species)

#comando que resume a distribui��o das classes, ou seja, quantos exemplos fazem parte de cada classe.
#os comandos retornam a quantidade e a porcentagem
percentage <- prop.table(table(dataset$Species)) * 100
cbind(freq=table(dataset$Species), percentage=percentage)
#da uma vis�o geral dos atributos, por exemplo: o atributo de valor m�ximo, m�nimo, m�dia...
summary(dataset)

# x � o conte�do da base de dados para os atributos de 1 at� 4
x <- dataset[,1:4]
# y � o conte�do do atributo classe
y <- dataset[,5]

# comando que gera um gr�fico para cada atributo
par(mfrow=c(1,4))
  for(i in 1:4) {
  boxplot(x[,i], main=names(iris)[i])
}

#comando que gera um gr�fico com a distribui��o das classes
plot(y)

#comando featurePlot gera gr�fico com a distribui��o dos dados
featurePlot(x=x, y=y, plot="ellipse") #n�o consegui rodar esse
featurePlot(x=x, y=y, plot="box")
featurePlot(x=x, y=y, plot="strip")
featurePlot(x=x, y=y, plot="density")
featurePlot(x=x, y=y, plot="pair")

#comando que indica o uso do cross validation igual a 10 e a m�trica acuracia
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# construindo modelos
# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)
#�summarize accuracy of models
results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)
# compare�accuracy of models
dotplot(results)

#para obter os resultados semelhantes ao do weka incluindo matriz de confus�o
# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)

#para acessar o tutorial de uma fun��o basta digitar ?nome_da_fun��o

#link material R 
#http://machinelearningmastery.com/machine-learning-in-r-step-by-step/
#http://machinelearningmastery.com/machine-learning-checklist/
#https://docs.microsoft.com/pt-br/azure/machine-learning/machine-learning-r-quickstart
#https://ocw.mit.edu/courses/sloan-school-of-management/15-097-prediction-machine-learning-and-statistics-spring-2012/lecture-notes/MIT15_097S12_lec02.pdf
