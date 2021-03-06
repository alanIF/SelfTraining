#PROBLEMAS A RESOLVER
#karliane e alan - aprender a usar outros classificadores que n?o seja arvore
#dividir a base em treinamento e teste, o q eu fiz n?o t? certo.

#1 - transformar os atributos n?o num?ricos em num?ricos - tentar filtro weka - alan achou paleativo, usaremos de acordo com a necessidade
#2 - descobrir pq a confian?a da iris s? d? 1 - resolvido, n?o sei como...

#bases de dados
#bupa, cleveland, ecoli, glass, haberman, iris, monk, pima, vehide, wisconsin
#diret?rio local para salvar as bases e resultados

#fazer a instalação/carregamento de pacotes e definir diretório local
source('C:/local_R/projeto_karliane/configuracoes.R')

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
  tx_g <- c()
  acc_g <- c()
  bd <- c()
  tx <- c()
#  acc <- 0.0
    
  for (i in 1:6){
  
    for (j in 1:5){      
      if (j == 1) taxa = 5
      else if (j == 2) taxa = 10
      else if (j == 3) taxa = 15
      else if (j == 4) taxa = 20
      else if (j == 5) taxa = 25
      
      source('C:/local_R/projeto_karliane/organiza_dados.R')
      print("iniciando o treinamento")
      
      #fun??o que ser? passada como par?metro predFunc da fun??o selftrain
      f <- function(m,d) {
        l <- predict(m,d,type='class')
        c <- apply(predict(m,d),1,max)
        data.frame(cl=l,p=c)
      }
      
      source('C:/local_R/projeto_karliane/treinamento.R')
      
    
  }
}
  #data frame que sera guardado no arquivo
  data_arquivo <- data.frame(tx_g,it_g,bd_g,thrConf_g,nr_added_exs_g)
  #escrever no arquivo
  write.csv(data_arquivo, "resultado.csv", row.names = FALSE)
  
  data_arquivo_acc <- data.frame(tx, bd, acc_g)
  write.csv(data_arquivo_acc, "resultado_acc.csv", row.names = FALSE)
  
