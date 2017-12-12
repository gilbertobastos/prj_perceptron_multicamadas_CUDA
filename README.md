# Perceptron Multicamadas

Implementação paralela do Perceptron Multicamadas na linguagem CUDA,
com o objetivo de comparar a performance e eficiência entre outras
implementações do Perceptron Multicamadas (OpenACC e sequencial), já
que o treinamento da rede é uma operação altamente custosa.

## A implementação

Como o foco do implementação não estava na eficiência do treinamento
da rede, e sim no na comparação de perfromance para meu TCC, a
implementação utiliza um modelo para o treinamento da mesma bem
simples, utilizando o algoritmo _backpropagation_, já que a rede pode
possuir mais de uma camada.

## Como compilar

Para compilar, basicamente serão necessários o __CUDA Toolkit__ (que
contém o compilador __nvcc___) e o nosso querido __GNU make__
(lembrando que todo código foi compilado utilizando um SO Linux, em
Windows não faço idéia do comportamento esperado :/ ) para fazer todo
o processo de compilação automatizado. Lembrando que no arquivo
__Makefile__ há diversas variáveis para configuração do código gerado,
como a arquitetura alvo e etc, onde o mesmo está configurado para
adaptadores gráficos com Compute Capability 6.1 (o código-fonte é
compatível com a grande maioria dos adaptadores mais antigos, pois não
usa operações atômicas e etc).

```sh
make # Executar este comando na pasta raíz do projeto (e torcer para compilar!)
```

## As etapas (básicas)  para o treinamento da rede

Há uma estrutura específica para os armazenar os padrões de
treinamento, onde a mesma consiste basicamente de dois vetores, um
vetor com a amostra à ser apresentada à rede e outra com a saída
desejada para rede:

```c
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
```

Essa estrutura, pode ser populada tanto manualmente, ou através de
dois arquivos no formato CSV (um com as amostras e outro com os
vetores de objetivo), utilizando a função abaixo (lembrando que os
dados serão carregados na memória do adaptador gráfico):

```c
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
```

Também é necessário alocar a estrutura do Perceptron Multicamadas na
memória utilizando a seguinte função:

```c
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
```

FINALMENTE, agora é a hora de iniciar o treinamento com a função abaixo (^_^):

```c
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
```

Para demais informações em relação as funções, basta olhar os arquivos
de cabeçalho, pois as funções estão devidamente documentadas (acredito
eu).
