#ifndef PERCEPTRON_MULTICAMADAS_KERNELS_CUH_
#define PERCEPTRON_MULTICAMADAS_KERNELS_CUH_

/* CUDA includes */
#include <cuda.h>
#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.h>
#undef __CUDA_INTERNAL_COMPILATION__
#include <vector_types.h>

/**
 * Enumerações para as funções de ativação da rede
 */
enum funcoes_ativacao_enum
{
  Identidade,
  Degrau,
  Sigmoide,
  Tang_Hiperbolica,
};

/**
 * Estrutura que representa uma camada da rede Perceptron Multicamadas.
 *
 * As variáveis com prefixo "d_" serão alocadas na memória do adaptador 
 * gráfico e as demais serão armazenadas no hospedeiro e quando necessário, 
 * serão copiadas automaticamente para o adaptador gráfico.
 */
typedef struct
{
  /** Vetor que irá armazenar os pesos desta camada na conveção
   * "row-major" (matriz). 
   */
  float * d_W;

  /** Vetor que irá armazenar o grau de ativação dos neurônios
   *  desta camada. 
   */
  float * d_n_ativacao;

  /** Vetor que irá armazenar a derivada da função de ativação dos neurônios
   * desta camada. 
   */
  float * d_n_derivada;

  /** Vetor que irá armazenar o erro retropropagado calculado para cada
   * neurônio desta camada. 
   */
  float * d_n_erro_rprop;

  /** Vetor que irá armazenar os bias para cada neurônio da camada. */
  float * d_v_bias;

  /** Variável que irá armazenar a quantidade de neurônios desta camada. */
  int qtd_neuronios;

  /** Variável que irá armazenar a função de ativação para esta camada
   * (usar a enumeração "FuncoesAtivacaoEnum"). */
  int funcao_ativacao;
  
} Camada;

/**
 * Estrutura que irá armazenar as camadas do Perceptron.
 */
typedef struct
{
  /** Vetor de referências para as camadas do Perceptron Multicamadas. */
  const Camada ** v_cam;

  /** Quantidade de camadas. */
  int qtd_camadas;

  /** Tamanho da entrada (quantidade de "neurônios"). */
  int qtd_neuronios_entrada;

} PerceptronMulticamadas;

/***********************************************************
 * Estruturas que irão armazenar as informações referentes *
 * ao treinamento da rede.                                 *
 ***********************************************************/

/**
 * Estrutura que irá armazenar as informações de 
 * uma época de treinamento da rede neural.
 */
typedef struct 
{
  /** Duração para o treinamento da época. */
  float duracaoSegs;

  /** Erro global para época após o treinamento
  da mesma. */
  float erroGlobal;
  
} InfoEpocaTreinamento;

/** Nó da lista... */
struct ListaNo
{
  InfoEpocaTreinamento dado;
  struct ListaNo * proxNo;
};

/**
 * Estrutura que irá armazenar informações
 * como a configuração da rede neural, taxa de 
 * aprendizagem e as informações sobre as 
 * épocas de treinamento.
 */
typedef struct
{
  /** Tamanho da entrada (camada de entrada). */
  int qtdNeuroniosEntrada;

  /** Quantidade de camadas da rede. */
  int qtdCamadas;

  /** Variável que irá armazenar a configuração
    * da rede neural em uma "string."
    *
    * Ex: 800-700-600-500
    */
  char strConfigRedeNeural[240];

  /** Taxa de aprendizagem da rede neural. */
  float taxaAprendizagem;

  /** Erro desejado. */
  float erroDesejado;

  /** Épocas de treinamento (linked list). */
  struct ListaNo * listaEpocas;
  
} HistoricoTreinamento;

/**
 * Método que realiza o cálculo da função degrau.
 *
 * @param z Parâmetro para o cálculo da função degrau.
 *
 * @return Valor do cálculo da função degrau.
 */
__device__ float funcao_degrau_dev(float z);

/**
 * Método que realiza o cálculo da derivada da função degrau.
 *
 * @param val_degrau Valor da função degrau da qual se deseja calcular
 *                   a derivada.
 *
 * @return Valor do cálculo da derivada da função degrau.
 */
__device__ float derivada_funcao_degrau_dev(float val_degrau);

/**
 * Método que realiza o cálculo da função sigmóide.
 *
 * @param z Parâmetro para calculo da sigmóide.
 *
 * @return Resultado do cálculo da funçãoo sigmóide.
 */
__device__ float funcao_sigmoide_dev(float z);

/**
 * Método que realiza o cálculo da derivada da função sigmóide, recebendo
 * por parâmetro o resultado do cálculo da função sigmóide.
 *
 * @param res_calc_sigmoide Resultado do cálculo da função sigmóide.
 *
 * @return Derivada da função sigmóide.
 */
__device__ float derivada_funcao_sigmoide_dev(float res_calc_sigmoide);

/**
 * Método que realiza o cálculo da função tangente hiperbólica.
 *
 * @param z Parâmetro para o cálculo da função tangente hiperbólica.
 *
 * @return Valor do cálculo da função tangente hiperbólica.
 */
__device__ float funcao_tang_hiperbolica_dev(float z);

/**
 * Método que realiza o cálculo da derivada da função tangente hiperbólica.
 *
 * @param val_tang_hiperbolica Valor da função tangente hiperbólica da qual se
 *                             deseja calcular a derivada.
 *
 * @return Valor do cálculo da derivada da função tangente hiperbólica.
 */
__device__ float derivada_funcao_tang_hiperbolica_dev(float val_tang_hiperbolica);

/**
 * Kernel que tem o objetivo de calcular a ativação dos neurônios que se
 * encontram na primeira camada.
 *
 * A função será descolada para o adaptador gráfico de forma unidimensional,
 * sendo assim, cada núcleo (SP ou CUDA Core) do adaptador gráfico irá realizar o
 * processamento de um neurônio.
 *
 * @param c Primeira camada da rede.
 *
 * @param d_amostra Amostra com seu conteúdo armazenado no adaptador gráfico.
 *
 * @param qtd_neuronios_entrada Quantidade de neuronios da camada de entrada.
 */
__global__ void funcao_ativacao_neuronio_primeira_camada_kernel
(const Camada c, const float * d_amostra, int qtd_neuronios_entrada);

/**
 * Kernel que tem o objetivo de calcular a ativação dos neurônios que se
 * encontram em qualquer camada da rede salvo a primeira.
 *
 * O funcionamento deste kernel é semelhante ao de cima (olhar o comentário
 * do kernel acima), apenas não será utilizada uma amostra e sim a ativação
 * dos neurônios da camada anterior a camada em que se deseja calcular a
 * ativação dos neurônios da mesma.
 *
 * @param c_ant Camada anterior a camada que se deseja calcular a ativação
 * dos neurônios.
 *
 * @param c Camada em que se deseja calcular a ativação dos neurônios.
 */
__global__ void funcao_ativacao_neuronio_camada_kernel(const Camada c_ant, const Camada c);

/**
 * Kernel que tem o objetivo de atualizar os pesos dos neurônios da primeira
 * camada do Perceptron Multicamadas.
 *
 * A divisão da tarefa para o adaptador gráfico será realizada de forma
 * semelhante as funções de integração.
 *
 * @param c Primeira camada da rede.
 *
 * @param d_amostra Amostra com seu conteúdo armazenado no adaptador gráfico.
 *
 * @param taxa_aprendizagem Taxa de aprendizagem.
 *
 * @param qtd_neuronios_entrada Quantidade de neurônios da camada de entrada.
 */
__global__ void funcao_atualizacao_pesos_neuronio_primeira_camada_kernel
(const Camada c, const float * d_amostra, float taxa_aprendizagem, int qtd_neuronios_entrada);

/**
 * Kernel que tem o objetivo de atualizar os pesos dos neurônios de alguma
 * camada da rede (salvo a primeira).
 *
 * A divisão da tarefa para o adaptador gráfico será realizada de forma
 * semelhante as funções de integração.
 *
 * @param c_ant Camada anterior a camada que se deseja realizar a atualização
 * dos pesos dos neurônios.
 *
 * @param c Camada em que se deseja realizar a atualização dos pesos
 * dos neurônios.
 *
 * @param taxa_aprendizagem Taxa de aprendizagem.
 */
__global__ void funcao_atualizacao_pesos_neuronio_camada_kernel
(const Camada c_ant, const Camada c, float taxa_aprendizagem);

/**
 * Kernel sequencial que tem o objetivo de calcular o erro retropropagado dos
 * neurônios da última camada da rede.
 *
 * Lembrando que a execução do kernel deve ser sequencial "<<<1, 1>>>"
 * pois o mesmo calcula o erro retropropagado de todos os neurônios e calcula
 * o erro global para o parâmetro apresentado a rede.
 *
 * @param c Camada (a última camada da rede).
 *
 * @param d_alvo Vetor com a saída desejada para os neurônios da última
 * camada.
 *
 * @param d_erro_padrao Variável que irá armazenar o erro calculado para o
 *                      padrão apresentado à rede.                   
 */
__global__ void funcao_calcular_erro_rprop_neuronio_ultima_camada_sequencial_kernel
(const Camada c, const float * d_alvo, float * d_erro_padrao);

/**
 * Kernel que tem o objetivo de calcular o erro retropropagado dos neurônios
 * de alguma camada da rede (salvo a última).
 *
 * A divisão da tarefa para o adaptador gráfico será realizada de forma
 * semelhante as funções de integração.
 *
 * @param c Camada a ter o erro retropropagado dos seus neurônios calculados.
 *
 * @param c_pos Camada posterior a camada acima.
 */
__global__ void funcao_calcular_erro_rprop_neuronio_camada_kernel
(const Camada c, const Camada c_pos);

#endif
