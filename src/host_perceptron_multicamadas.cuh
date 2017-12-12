/****************************************************************************
 * Projeto Perceptron Multicamadas.                                         *
 *                                                                          *
 * Implementação da rede Perceptron Multicamadas com treinamento utilizando *
 * o método "backpropagation" para classificação das amostras em CUDA.      *
 *                                                                          *
 * @author Gilberto Augusto de Oliveira Bastos.                             *
 * @copyright BSD-2-Clause                                                  *
 ****************************************************************************/

#ifndef HOST_PERCEPTRON_MULTICAMADAS_CUH_
#define HOST_PERCEPTRON_MULTICAMADAS_CUH_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <stdbool.h>
#include <complex.h>
#include <cuda_runtime_api.h>
#include "uniform.h" /* Biblioteca para gerar números aleatórios uniformemente
                        distribuídos. */

/** Para geração dos arquivos de histórico de treinamento. */
#define DELIMITADOR_CSV ','

/* Cabeçalhos dos kernels e funções do dispositivo e
 * estruturas (structs) do Perceptron Multicamadas e
 * do Histórico de Treinamento...
 */
#include "perceptron_multicamadas_kernels.cuh"

/** Quantidade máxima de épocas de treinamento. */
#define QTD_MAX_EPOCAS 1000

/** Quantidade de threads que cada grupo deverá ter. */
#define QTD_THREADS_GRUPO 32 //Pascal GP107 -> 128 núcleos por SM

/** Intervalos para geração dos números randômicos
 para os pesos. */
#define LIM_MIN -1
#define LIM_MAX  1

/** Bias... */
#define BIAS 1.0

/* Para mostra informações estatísticas. */
#define INFO_ESTATISTICAS 1

/**
 * Estrutura que irá representar um padrão de treinamento
 * para ser apresentado à rede.
 */
typedef struct {
  /** A vetor com a amostra (entrada da rede). */
  const float * d_amostra;
  
  /** Vetor com os valores desejados para saída da rede (objetivo). */
  const float * d_alvo;

} PadraoTreinamento;

/**
 * Método que realiza o treinamento da rede através do método "backpropagation"
 * utilizando os padrões passados por parâmetro. A rede irá ser treinada
 * até que o erro MSE da mesma seja menor ou igual ao "erro_desejado" OU
 * até que a rede atinja a quantida máxima de épocas (QTD_MAX_EPOCAS).
 *
 * @param pm Perceptron.
 *
 * @param padroes Padrões para treinamento.
 *
 * @param qtd_padroes_treinamento Quantidade de padrões de treinamento.
 *
 * @param taxa_aprendizagem Taxa de aprendizagem.
 *
 * @param erro_desejado Condição de parada para o treinamento
 * da rede.
 *
 * @param gerar_historico Se será necessário gerar o histórico ou não.
 *
 * @return Histórico de treinamento.
 */
HistoricoTreinamento *
treinamento_PerceptronMulticamadas(PerceptronMulticamadas * pm,
				   PadraoTreinamento * padroes,
				   int qtd_padroes_treinamento,
				   float taxa_aprendizagem,
				   float erro_desejado,
				   bool gerar_historico);
/**
 * Método que tem o objetivo de realizar a configuração da disposição das
 * threads e dos grupos de um kernel.
 *
 * @param c Camada sobre a qual se deseja realizar a divisão de trabalho.
 *
 * @param blocksGrid Variável que irá armazenar a configuração dos blocos
 * para a camada.
 *
 * @param threadsBlock Variável que irá armazenar a configuração das threads
 * para a camada (de forma bidimensional).
 */
void obter_configuracao_threads_1D(const Camada c, dim3 * gridConfig,
				   dim3 * blockConfig);

/**
 * Método que realiza a normalização de um vetor através do método "min-max".
 *
 * @param v Vetor a ser embaralhado.
 *
 * @param n Quantidade de itens do vetor.
 *
 * @param min Mínimo.
 *
 * @param max Máximo.
 */
void normalizacao_min_max(float * v, int n, float min, float max);

/**
 * Método que realiza o embaralhamento de um vetor de inteiros através
 * do método de Fisher-Yates (moderno).
 *
 * @param v Vetor a ser embaralhado.
 *
 * @param n Quantidade de itens do vetor.
 */
void embaralhamento_Fisher_Yates(int * v, int n);

/**
 * Método que carrega os padrões de treinamento de dois arquivos, um com as
 * amostras (que são normalizadas pela função), onde cada linha do mesmo
 * representa uma amostra (com os valores separados por ponto e vírgula ";"), e
 * outro com o vetor de objetivos para cada amostra respectivamente
 * (com os valores separados por ponto e vírgula também).
 *
 * @param nome_arquivo_amostras Nome do arquivo (com extensão) com as amostras
 *                              dos padrões de treinamento ou de teste.
 *
 * @param nome_arquivo_objetivos Nome do arquivo (com extensão) com os
 *                               objetivos dos padrões de treinamento ou
 *                               de teste.
 *
 * @param menor_val_amostra Menor valor presente nas amostras
 *                          (para normalização)
 *
 * @param maior_val_amostra Maior valor presente nas amostras
 *                          (para normalização)
 *
 * @param qtd_itens_amostra Quantidade de itens por amostra.
 *
 * @param qtd_itens_v_objetivo Quantidade de itens por vetor de objetivo.
 *
 * @param qtd_padroes Quantidade de padrões nos arquivos.
 *
 * @return Vetor com os padrões de treinamento ou de teste carregados na memória
 *         do adaptador gráfico ou NULO caso não seja possível abrir os arquivos
 *         para leitura.
 */
PadraoTreinamento * carregar_padroes_arquivo(char * nome_arquivo_amostras,
		                             char * nome_arquivo_objetivos,
					     float menor_val_amostra,
					     float maior_val_amostra,
					     int qtd_itens_amostra,
					     int qtd_itens_v_objetivo,
					     int qtd_padroes);

/**
 * Método que aloca o Perceptron Multicamadas na memória do hospedeiro,
 * salvo os atributos das camadas que serão salvos no adaptador gráfico.
 *
 * @param qtd_neuronios_entrada Quantidade de neurônios da camada de
 *                              entrada.
 *
 * @param qtd_camadas Quantidade de camadas (em que há processamento).
 *
 * @param qtd_neuronios_camada Vetor com a quantidade de neuronios para cada
 *                             camada.
 * 
 * @param funcao_ativacao_rede Função de ativação da rede, ou seja, todas as
 *                             camadas da rede estarão atribuidas para serem
 *                             ativadas com tal função (usar a enumeração
 *                             "funcoes_ativacao_enum"). Lembrando que, caso se
 *                             deseje que as camadas tenham funções de ativação
 *                             diferente, estabelecer manualmente estes valores
 *                             nas respectivas camadas. 
 *
 * @return Referência para a estrutura alocada.
 */
PerceptronMulticamadas *
inicializar_PerceptronMulticamadas(int qtd_neuronios_entrada,
				   int qtd_camadas,
				   int * qtd_neuronios_camada,
				   int funcao_ativacao_rede);

/**
 * Método que aloca na memória do hospedeiro a camada do Perceptron
 * Multicamadas (os atributos da mesma serão salvos no adaptador gráfico).
 *
 * @param qtd_neuronios Quantidade de neurônios que a camada irá possuir.
 *
 * @param qtd_pesos_neuronio Quantidade de pesos que cada
 * neurônio irá possuir.
 *
 * @param funcaoAtivacao Função de ativação desta camada
 *                       (usar a enumeração "funcoes_ativacao_enum").
 *
 * @return Referência para a camada alocada.
 */
Camada * __alocar_camada(int qtd_neuronios, int qtd_pesos_neuronio,
			 int funcao_ativacao);

/**
 * Método que aloca um vetor de pesos na memória do hospedeiro, gera os números
 * aleatórios (intervalo de LIM_MIN, LIM_MAX) e retorna a referência para o
 * mesmo.
 *
 * @param qtd_pesos Quantidade de pesos do a serem alocados.
 *
 * @return Referência para o vetor de pesos alocado na memória.
 */
float * __alocar_vetor_pesos_randomicos(int qtd_pesos);

/**
 * Método que realiza alimentação da rede (feedfoward) com amostra.
 *
 * @param h_pm Referência para Perceptron Multicamadas.
 *
 * @param d_amostra Vetor da amostra (que deve estar alocado na memória do
 * adaptador gráfico).
 */
void alimentar_PerceptronMulticamadas(PerceptronMulticamadas * pm,
				      const float * d_amostra);

/**
 * Método que calcula a taxa de acerto de uma rede Perceptron Multicamdas já
 * treinada utilizando os padrões de teste.
 *
 * @param pm Referência para o Perceptron Multicamadas.
 *
 * @param padroes_teste Padrões de teste (devem utilizar a a mesma estrutura
 *                      dos padrões de treinamento).
 *
 * @param qtd_padroes_teste Quantidade de padrões de teste.
 *
 * @return Taxa de acerto da rede (erro MSE).
 */
float calcular_taxa_acerto_PerceptronMulticamdas(PerceptronMulticamadas * pm,
						 PadraoTreinamento * padroes_teste,
						 int qtd_padroes_teste);

/**
 * Método que tem o objetivo de inicializar a estrutura do
 * histórico de treinamento. 
 *
 * @param pm Referência para o Perceptron Multicamadas.
 *
 * @param taxaAprendizagem Taxa de aprendizagem da rede.
 *
 * @param erroDesejado Erro desejado para que seja encerrado o
 *                     treinamento.
 *
 * @return Referência para a estrutura alocada.
 */
HistoricoTreinamento *
HistoricoTreinamento_inicializar(PerceptronMulticamadas * pm,
				 float taxaAprendizagem,
				 float erroDesejado);

/**
 * Método que tem o objetivo de adicionar as informações
 * de uma época de treinamento de uma rede neural no 
 * histórico.
 *
 * @param historicoTreinamento Estrutura do histórico de treinamento.
 *
 * @param duracaoSegs Duração para o treinamento da 
 *                    época em segundos.
 *
 * @param erroGlobal Erro global para época após o treinamento
 *                   da mesma.
 */
void HistoricoTreinamento_adicionarInfoEpoca(HistoricoTreinamento * 
					     historicoTreinamento,
					     float duracaoSegs,
					     float erroGlobal);

/**
 * Método que cria um arquivo *.csv com todo histórico das 
 * épocas.
 *
 * A primeira linha do arquivo irá conter a arquitetura da rede,
 * a quantidade de padrões de treinamento, taxa de aprendizagem e o
 * erro desejado respectivamente.
 * As demais linhas irão conter o número da época, duração da época
 * em segundos e por fim o erro global após o treinamento da época.
 * 
 * @param historicoTreinamento Estrutura do histórico de treinamento.
 *
 * @param nomeArquivo Nome do arquivo *.csv a ser gerado.
 */
void HistoricoTreinamento_gerarArquivoCSV(HistoricoTreinamento * 
					  historicoTreinamento,
					  char * nomeArquivo);

#endif

