#PROBLEMAS A RESOLVER
#karliane - tentar pegar os resultados (acuracia) - ver base teste script alexandre
#alan - aprender como colocar os resultados em uma matriz e depois em um arquivo
#alan - incluir as demais bases nesse script
#karliane e alan - aprender a usar outros classificadores que não seja arvore
#1 - transformar os atributos não numéricos em numéricos - tentar filtro weka - alan achou paleativo, usaremos de acordo com a necessidade
#2 - descobrir pq a confiança da iris só dá 1 - resolvido, não sei como...

#bases de dados
#,  vehide, wisconsin
#diretório local para salvar as bases e resultados
setwd("C:\\local_R")

print("instalação dos pacotes")

#pacote que inclui: data splitting, pre-processing, feature selection, model tuning using resampling, variable importance estimation
#install.packages("caret")
#install.packages("caret", dependencies = c("Depends", "Suggests"))
#pacote que inclui self-training e outros algoritmos de aprendizado semisupervisionado
#install.packages("ssc")
#install.packages("DMwR")
#install.packages("caTools")
#install.packages("RWeka")

print("carregar os pacotes")
library("caret") #parece não ser necessário
library("ssc") #esse é obrigatório
library("plyr") #pacote q tem a função join_all
library("RWeka")

#USANDO A FUNÇÃO SELFTRAIN (USADA POR ALEXANDRE)

library("DMwR2")
library("DMwR")
library("datasets")

print("Função para pegar a base de dados e colocar em uma variável base")
getdata <- function(...)
{
    e <- new.env()
    name <- data(..., envir = e)[1]
    e[[name]]
}

#variaveis para guardar e gravar no arquivo
it_g <-c()
bd_g <-c()
thrConf_g<-c()
nr_added_exs_g<-c()

for (i in 1:7){

  print("organizando os dados")

  if (i==1) {
    #base de dados IRIS
    base <- getdata("iris")
    classe <- "Species"
  }else if (i==2){
    #base de dados ECOLI
    base <- read.arff("ecoli.arff")
    classe <- "class"
  }
  else if(i==3){
    base <- read.arff("bupa.arff");
    classe <- "selector"
    
  }
  else if(i==4){
    base <- read.arff("glass.arff")
    classe <- "Type"
    
  }
  else if(i==5){
    base <- read.arff("haberman.arff")
    classe <-"Survival_status"
  }
  else if(i==6){
    base <-read.arff("pima.arff")
    classe <- "class"
    
  }
  else if(i==7){
    base <-read.arff("cleveland.arff")
    classe <- "num"
    
  }
  #tentando usar filtro do weka para transformar dados nominais em binarios
  #nombi <- make_Weka_filter("weka/filters/supervised/attribute/NominalToBinary") # creates an R interface to the WEKA filter
  #datbin <- nombi(AT1 ~., data=base, control =Weka_control(N=TRUE, A=TRUE)) # Fehlermeldung
  #datbin
  
  
  
  set.seed(214)# garante que o conjunto de dados escolhido para treinamento será sempre o mesmo - não sei se preciso dessa garantia
  
  #Quantidade de Exemplos
  exemplos = nrow(base)
  
  #taxa inicial de exemplos rotulados erm percentual
  taxa = 10
  taxainicial = exemplos*taxa/100
  
  #sorteio de ids para treinamento
  ids_treino_rot <- sample(exemplos,taxainicial, replace=FALSE)

  #base de treinamento
  basetreinorot <- base[ids_treino_rot,]
  basetreinosemrot <- base[-ids_treino_rot,]


  if (i==1) basetreinosemrot$Species <- NA #para base IRIS
  else if (i==2) basetreinosemrot$class <- NA #para base ECOLI
  else if(i==3)  basetreinosemrot$selector <- NA # para base BUPA
  else if(i==4)   basetreinosemrot$Type <- NA # para base glass
  else if(i==5) basetreinosemrot$Survival_status<- NA # para base haberman
  else if (i==6) basetreinosemrot$class <- NA #para base pima
  else if (i==7) basetreinosemrot$num <- NA #para base cleveland
  
  #base de treinamento rotulada
  basetreinoselftrainingrot <- basetreinorot
  basetreinoselftrainingsemrot <- basetreinosemrot
  dfs <- list(basetreinoselftrainingrot, basetreinoselftrainingsemrot)
  basetreinoselftraining <- join_all(dfs, type="full")
  
  print("iniciando o treinamento")
  #função que será passada como parâmetro predFunc da função selftrain
  f <- function(m,d) {
  	l <- predict(m,d,type='class')
  	c <- apply(predict(m,d),1,max)
  	data.frame(cl=l,p=c)
  }
  
  #setando parametros do selftrain
  
  #classes da base de dados
  if (i==1) form <- Species~.  	#para base IRIS		  #OU form <- basetreinoselftraining$Species
  if (i==2) form <- class~.      #para base ECOLI
  if(i==3) form <- selector~.    #para base puma
  if(i==4)form <- Type~. # para base glass
  if(i==5)form <- Survival_status~.# base haberman
  if (i==6) form <- class~.      #para base pima
  if(i==7) form <- num~. # para base cleveland
  data <- basetreinoselftraining	#base de dados
  learn <- learner('rpartXse',list(se=0.5))
  predFunc <- 'f'   			#Uma string com o nome de uma função que irá realizar as tarefas de classificação probabilística que serão necessárias durante o processo de self-training
  thrConf=0.9       			#taxa de confiança dos exemplos a serem incluidos no conjunto de rotulados
  maxIts=10					#número máximo de iterações
  percFull=1					#Um número entre 0 e 1. Se a porcentagem de exemplos rotulados atingir esse valor o processo de self-training é parado
  verbose=TRUE				#Um booleano indicando o nível de verbosidade?? (verbosity??) da função
  
  #adaptação da implementação do selftrain
  data
  N <- NROW(data)
  it <- 0
  
  
  somaConf <- 0
  qtdExemplosRot <- 0
  totalrot <- 0
  
  sup <- which(!is.na(data[,as.character(form[[2]])])) #sup recebe o indice de todos os exemplos rotulados
      repeat {
        
        it <- it+1
  	

      	if (it>1) thrConf <- (thrConf + (somaConf/qtdExemplosRot) + (qtdExemplosRot/N))/3
      	somaConf <- 0
      	qtdExemplosRot <- 0

  
        model <- runLearner(learn,form,data[sup,])
        probPreds <- do.call(predFunc,list(model,data[-sup,]))

  
        new <- which(probPreds[,2] > thrConf)

  
        

        if (verbose) {
            cat('IT.',it,'BD',i,thrConf,'\t nr. added exs. =',length(new),'\n') 
            ##guardando nas variaveis 
            it_g <-c(it_g,it)
            bd_g <-c(bd_g,i)
            thrConf_g<-c(thrConf_g,thrConf)
            nr_added_exs_g<-c(nr_added_exs_g,length(new))
            ##resultado <-  c(it,",",i,",",thrConf,",",length(new))
            ##write(resultado, file = "result")
          
        }

        if (length(new)) {
          data[(1:N)[-sup][new],as.character(form[[2]])] <- as.character(probPreds[new,1])
  
  	      somaConf <- sum(somaConf, probPreds[new,2])
  	      qtdExemplosRot <- length(data[(1:N)[-sup][new],as.character(form[[2]])])
  	      totalrot <- totalrot + qtdExemplosRot

          sup <- c(sup,(1:N)[-sup][new])
        } else break
        if (it == maxIts || length(sup)/N >= percFull) break
      }

  cat('FIM', '\t base de dados ', i, '\n', 'total rotulados: ', totalrot, '\n')
}
#data frame que sera guardado no arquivo
data_arquivo <- data.frame(it_g,bd_g,thrConf_g,nr_added_exs_g)
#escrever no arquivo
write.csv(data_arquivo, "resultado.csv", row.names = FALSE)
