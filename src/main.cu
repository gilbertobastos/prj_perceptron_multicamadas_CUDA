#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "host_perceptron_multicamadas.cuh"

int main(void)
{
  /* Carregando os padrões XOR para treinamento
     da rede. */  
  PadraoTreinamento * padroes_XOR =
    carregar_padroes_arquivo(
      "./dataset-xor-amostras.csv",
      "./dataset-xor-objetivos.csv", 0, 1, 3, 1, 4);

  /* Criando a rede... */
  PerceptronMulticamadas * h_pm;
  int qtd_neuronios_camada[] = {2, 1};
  h_pm = inicializar_PerceptronMulticamadas(3, 2, qtd_neuronios_camada, Sigmoide);

  /* Treinando a rede e criando um histórico com as
     informações do treinamento. */
  HistoricoTreinamento * historicoTreinamento;
  historicoTreinamento =
    treinamento_PerceptronMulticamadas(h_pm, padroes_XOR,
				       4, 0.001,
				       0.0010, true);

  /* Salvando histórico de treinamento em um arquivo. */
  char nomeArquivoHistorico[] = "historico_treinamento_XOR_CUDA.csv";
  HistoricoTreinamento_gerarArquivoCSV(historicoTreinamento, nomeArquivoHistorico);
  
  return 0;
}
