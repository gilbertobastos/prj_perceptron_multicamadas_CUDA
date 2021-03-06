#include "host_perceptron_multicamadas.cuh"

PerceptronMulticamadas *
inicializar_PerceptronMulticamadas(int qtd_neuronios_entrada,
				   int qtd_camadas,
				   int * qtd_neuronios_camada,
				   int funcao_ativacao_rede)
{
  /* Criando o vetor de referências que irá armazenar as camadas. */
  const Camada ** v_cam = (const Camada**) malloc(sizeof(Camada *) * qtd_camadas);

  /* Alocando as camadas na memória (lembrando que os atributos com o prefixo
   * "d_" serão alocados diretamente na memória do adaptador gráfico). */
  v_cam[0] = __alocar_camada(qtd_neuronios_camada[0], qtd_neuronios_entrada,
			     funcao_ativacao_rede);
  
  for (int i = 1; i < qtd_camadas; i++)
  {
    v_cam[i] = __alocar_camada(qtd_neuronios_camada[i],
			       qtd_neuronios_camada[i - 1],
			       funcao_ativacao_rede);
  }

  /* Alocando a estrutura que irá abrigar as camadas. */
  PerceptronMulticamadas * pm =
    (PerceptronMulticamadas *) malloc(sizeof(PerceptronMulticamadas));

  /* Preenchendo os atributos do Perceptron. */
  pm->v_cam = v_cam;
  pm->qtd_camadas = qtd_camadas;
  pm->qtd_neuronios_entrada = qtd_neuronios_entrada;

  /* Retornando a estrutura do Perceptron Multicamadas. */
  return pm;
}

Camada * __alocar_camada(int qtd_neuronios, int qtd_pesos_neuronio,
			 int funcao_ativacao)
{
  /* Estrutura da camada. */
  Camada * c = (Camada *) malloc(sizeof(Camada));

  /* Gerando o vetor com os pesos aleatórios no hospedeiro. */
  float * h_vetor_pesos = __alocar_vetor_pesos_randomicos(
			qtd_pesos_neuronio * qtd_neuronios);

  /* Alocando o espaço para o vetor de pesos no adaptador gráfico e
   * copiando o vetor de pesos randômicos gerados no hospedeiro
   * para o adaptador gráfico.
   */
  cudaMalloc((void **) &c->d_W, sizeof(float) *
	     (qtd_pesos_neuronio * qtd_neuronios));
  cudaMemcpy(c->d_W, h_vetor_pesos, sizeof(float) *
	     (qtd_pesos_neuronio * qtd_neuronios), cudaMemcpyHostToDevice);

  /* Alocando no adaptador gráfico o vetor que irá armazenar a ativação
   * dos neurônios.
   */
  cudaMalloc((void **) &c->d_n_ativacao, sizeof(float) * qtd_neuronios);

  /* Alocando no adaptador gráfico o vetor que irá armazenar as derivadas
   * dos neurônios.
   */
  cudaMalloc((void **) &c->d_n_derivada, sizeof(float) * qtd_neuronios);
  
  /* Alocando no adaptador gráfico o vetor que irá armazenar o erro calculado
   * retropropagado para cada neurônio.
   */
  cudaMalloc((void **) &c->d_n_erro_rprop, sizeof(float) * qtd_neuronios);

  /* Alocando o vetor de bias no hospedeiro e preenchendo o mesmo
   * com o valor da macro BIAS. */
  float * h_v_bias;
  h_v_bias = (float *) malloc(sizeof(float) * qtd_neuronios);
  
  for (int i = 0; i < qtd_neuronios; i++)
  {
    h_v_bias[i] = BIAS;
  }

  /* Alocando o espaço para o vetor de bias no adaptador gráfico
   * e copiando o vetor de bias do hospedeiro para o mesmo.
   */
  cudaMalloc((void **) &c->d_v_bias, sizeof(float) * qtd_neuronios);
  cudaMemcpy(c->d_v_bias, h_v_bias, sizeof(float) * qtd_neuronios,
	     cudaMemcpyHostToDevice);

  /* Preenchendo os demais atributos. */
  c->qtd_neuronios = qtd_neuronios;
  c->funcao_ativacao = funcao_ativacao;

  /* Desalocandos os vetores que já foram copiados para o adaptador gráfico
   * que estão no hospedeiro.
   */
  free(h_v_bias);
  free(h_vetor_pesos);

  /* Retornando a referência para a camada alocada no adaptador
   * gráfico.
   */
  return c;
}

float * __alocar_vetor_pesos_randomicos(int qtd_pesos)
{  
  /* Alocando o vetor de pesos. */
  float * vetor_pesos;
  vetor_pesos = (float *) malloc(sizeof(float) * qtd_pesos);

  /* Gerando a semente (número entre 0 a 99 999)... */
  srand(time(NULL));
  int semente = rand() % 100000;

  for (int i = 0; i < qtd_pesos; i++)
  {
    /* Gerando o peso no intervalo de LIM_MIN..LIM_MAX. */
    vetor_pesos[i] = r4_uniform_ab(LIM_MIN, LIM_MAX, &semente);
  }

  /* Retornando a referência para o vetor alocado. */
  return vetor_pesos;
}

void normalizacao_min_max(float * v, int n, float min, float max)
{
  /* Percorrendo todos os itens do vetor e realizando a normalização
   * dos mesmos. */
  for (int i = 0; i < n; i++)
  {
    v[i] = (v[i] - min) / (max - min);
  }
}

void embaralhamento_Fisher_Yates(int * v, int n)
{
  /* Gerando a semente para gerar
   * os números randômicos. */
  srand(time(NULL));

  /* Percorrendo o vetor do fim para o
   * início. */
  for (int i = n - 1; i > 0; i--)
  {
    /* Gerando o índice para permutação
     * entre o número "i-ésimo" e algum
     * número entre 0 e i-1. */
    int indice_perm = rand() % i;

    /* Realizando a troca. */
    int aux_troca = v[i];
    v[i] = v[indice_perm];
    v[indice_perm] = aux_troca;
  }
}

void obter_configuracao_threads_1D
(const Camada c, dim3 * gridConfig, dim3 * blockConfig)
{
  /* Calculando a quantidade de grupos que devem ser criados
   * para a camada.
   */
  gridConfig->x = (int) ceilf((float) c.qtd_neuronios / QTD_THREADS_GRUPO);
  gridConfig->y = 1;
  gridConfig->z = 1;

  /* Configurando as dimensões do grupo de threads... */
  blockConfig->x = QTD_THREADS_GRUPO;
  blockConfig->y = 1;
  blockConfig->z = 1;
}

void alimentar_PerceptronMulticamadas(PerceptronMulticamadas * pm, const float * d_amostra)
{
  /* Variáveis que irão auxíliar na configuração da disposição das threads
   * para os kernels.
   */
  dim3 blocksGrid;
  dim3 threadsBlock;

  /* Calculando o valor da função de integração e ativação
   * para os neurônios da primeira camada da rede.
   */
  obter_configuracao_threads_1D(*pm->v_cam[0], &blocksGrid, &threadsBlock);

  funcao_ativacao_neuronio_primeira_camada_kernel<<<blocksGrid, threadsBlock>>>
    (*pm->v_cam[0], d_amostra, pm->qtd_neuronios_entrada);

  /* Calculando o valor das demais camadas. */
  for (int camada = 1; camada < pm->qtd_camadas; camada++)
  {
    obter_configuracao_threads_1D(*pm->v_cam[camada], &blocksGrid,
				  &threadsBlock);

    funcao_ativacao_neuronio_camada_kernel<<<blocksGrid, threadsBlock>>>
      (*pm->v_cam[camada - 1], *pm->v_cam[camada]);
  }
}

HistoricoTreinamento *
treinamento_PerceptronMulticamadas(PerceptronMulticamadas * pm,
				   PadraoTreinamento * padroes,
				   int qtd_padroes_treinamento,
				   float taxa_aprendizagem,
				   float erro_desejado,
				   bool gerar_historico)
{
  /* Inicializando a estrutura. */
  HistoricoTreinamento * historicoTreinamento; 
  if (gerar_historico)
  {
    historicoTreinamento = HistoricoTreinamento_inicializar(pm,
							    taxa_aprendizagem,
							    erro_desejado);
  }
  
  /* Variáveis que irão auxíliar na configuração da disposição das threads
   * para os kernels.
   */
  dim3 blocksGrid;
  dim3 threadsBlock;
  
  /* Variável que irá armazenar o erro global calculado para cada iteração.
   * A mesma será populada após a execução de uma iteração, onde será copiado
   * o erro global que está na memória do adaptador gráfico para o hospedeiro.
   */
  float h_erro_global;
  
  /* Alocando na memória do adaptador gráfico a variável que irá armazenar
   * o erro para os padrões apresentado à rede.
   */
  float * d_erro_padrao;
  cudaMalloc((void **) &d_erro_padrao, sizeof(float));
  
  /* O treinamento irá ocorrer enquanto o erro da rede estiver acima
   * do desejado OU a quantidade de épocas não tenha atingido o limite. */
  int epocas = 0;
  
  do {
    /* Zerando a variável que irá armazenar o erro global para 
     * a iteração. */
    h_erro_global = 0;

    /* Variáveis que serão utilizadas para armazenar a hora que foi iniciada
       e finalizada o treinamento da rede para a época atual. */
    cudaEvent_t inicio_treinamento_epoca;
    cudaEvent_t fim_treinamento_epoca;
    cudaEventCreate(&inicio_treinamento_epoca);
    cudaEventCreate(&fim_treinamento_epoca);

    /* Coletando a hora antes do início do treinamento para época atual. */
    cudaEventRecord(inicio_treinamento_epoca);

    /* Apresentando os padrões de treinamento para rede e realizando o
     * treinamento da mesma. */
    for (int i = 0; i < qtd_padroes_treinamento; i++)
    {
      /* Alimentando a rede com o padrão "i-ésimo". */
      alimentar_PerceptronMulticamadas(pm, padroes[i].d_amostra);

      /* Calculando o erro dos neurônios da última camada. */
      funcao_calcular_erro_rprop_neuronio_ultima_camada_sequencial_kernel<<<1, 1>>> // <- Kernel sequencial
	(*pm->v_cam[pm->qtd_camadas - 1], padroes[i].d_alvo, d_erro_padrao);

      /* Copiando o erro do padrão armazenado no adaptador gráfico para o hospedeiro. */
      float h_erro_padrao;
      cudaMemcpy(&h_erro_padrao, d_erro_padrao, sizeof(float),
		 cudaMemcpyDeviceToHost);
      
      /* Somando o erro calculado para o padrão no erro global. */
      h_erro_global += h_erro_padrao;
      
      /* Realizando a retropropagação do erro para as demais camadas. */
      for (int camada = pm->qtd_camadas - 2; camada >= 0; camada--) {
	obter_configuracao_threads_1D(*pm->v_cam[camada], &blocksGrid,
				      &threadsBlock);

	funcao_calcular_erro_rprop_neuronio_camada_kernel<<<blocksGrid, threadsBlock>>>
	  (*pm->v_cam[camada], *pm->v_cam[camada + 1]);
      }

      /* Atualizando os pesos dos neurônios da primeira camada. */
      obter_configuracao_threads_1D(*pm->v_cam[0], &blocksGrid, &threadsBlock);

      funcao_atualizacao_pesos_neuronio_primeira_camada_kernel<<<blocksGrid, threadsBlock>>>
	(*pm->v_cam[0], padroes[i].d_amostra, taxa_aprendizagem, pm->qtd_neuronios_entrada);

      /* Atualizando os pesos dos neurônios das demais camadas. */
      for (int camada = 1; camada < pm->qtd_camadas; camada++)
	{
	  obter_configuracao_threads_1D(*pm->v_cam[camada], &blocksGrid,
					&threadsBlock);
	
	  funcao_atualizacao_pesos_neuronio_camada_kernel<<<blocksGrid,threadsBlock>>>
	    (*pm->v_cam[camada - 1],*pm->v_cam[camada], taxa_aprendizagem);
	}
    }

    /* Coletando a hora depois do treinamento para a época atual. */
    cudaEventRecord(fim_treinamento_epoca);
    cudaEventSynchronize(fim_treinamento_epoca);

    /* Realizando o cálculo do MSE. */
    h_erro_global /= qtd_padroes_treinamento;

    /* Atualizando a quantidade de épocas. */
    epocas++;

    /* Calculando o tempo de treinamento. */
    float milisegs;
    cudaEventElapsedTime(&milisegs, inicio_treinamento_epoca,
			   fim_treinamento_epoca);
    float segs = milisegs / 1000.0;

    /* Adicionando as informações desta época no histórico de
    treinamento. */
    if (gerar_historico)
    {
      HistoricoTreinamento_adicionarInfoEpoca(historicoTreinamento,
					      segs,
					      h_erro_global);
    }
    
    /* Mostrando a época e o erro MSE para a mesma. */
    if (INFO_ESTATISTICAS)
    {
      printf("Época: %d\nErro MSE: %.4f\n\n", epocas, h_erro_global);
      printf("Tempo total de execução da época: %.2f segundo(s)\n\n", segs);
    }
    
  } while (h_erro_global > erro_desejado && epocas < QTD_MAX_EPOCAS);

  /* Desalocando variáveis que não são mais necessárias da memória
   * do adaptador gráfico.
   */
  cudaFree(d_erro_padrao);

  /* Retornando a quantidade de épocas... */
  return historicoTreinamento;
}

PadraoTreinamento * carregar_padroes_arquivo(char * nome_arquivo_amostras,
					     char * nome_arquivo_objetivos,
					     float menor_val_amostra,
					     float maior_val_amostra,
					     int qtd_itens_amostra,
					     int qtd_itens_v_objetivo,
					     int qtd_padroes)
{
  /* Tentando abrir os arquivos para leitura. */
  FILE * arq_amostras;
  FILE * arq_v_objetivos;

  arq_amostras = fopen(nome_arquivo_amostras, "r");
  arq_v_objetivos = fopen(nome_arquivo_objetivos, "r");

  /* Verificando se os arquivos foram abertos com sucesso. */
  if (arq_amostras == NULL || arq_v_objetivos == NULL)
  {
    /* Retornando nulo. */
    return NULL;
  }

  /* Alocando o vetor que irá armazenar os padrões. */
  PadraoTreinamento * padroes;
  padroes = (PadraoTreinamento *) malloc(sizeof(PadraoTreinamento) * qtd_padroes);

  /* Primeiramente lendo as amostras e inserindo as mesmas nos respectivos
   * padrões. */
  for (int i = 0; i < qtd_padroes; i++)
  {
    /* Coletando a linha com a amostra do arquivo. */
    char linha_amostra[8192]; // 8 Kbytes
    fscanf(arq_amostras, "%s", linha_amostra);

    /* Alocando o vetor para armazenar a amostra "i-ésima" no hospedeiro. */
    float amostra[qtd_itens_amostra];

    /* Extraindo o primeiro item da amostra. */
    amostra[0] = atof(strtok(linha_amostra, ";\n\0"));

    /* Extraindo os demais itens da amostra. */
    for (int j = 1; j < qtd_itens_amostra; j++)
   {
     /* Extraindo o item "j-ésimo" da amostra. */
     amostra[j] = atof(strtok(NULL, ";\n\0"));
   }

    /* Normalizando a amostra coletada... */
    normalizacao_min_max(amostra, qtd_itens_amostra, menor_val_amostra,
			 maior_val_amostra);

    /* Copiando o vetor no hospedeiro da amostra normalizada para a memória
     * o adaptador gráfico.
     */
    float * d_amostra;
    cudaMalloc((void **) &d_amostra, sizeof(float) * qtd_itens_amostra);
    cudaMemcpy(d_amostra, amostra, sizeof(float) * qtd_itens_amostra,
	       cudaMemcpyHostToDevice);

    /* Por fim, colocando no padrão a amostra acima extraida
     * do arquivo. */
    padroes[i].d_amostra = d_amostra;
  }

  /* Lendo os vetores de objetivo e inserindo os mesmos nos respectivos
   * padrões. */
  for (int i = 0; i < qtd_padroes; i++)
  {
    /* Coletando a linha com o vetor de objetivo do arquivo. */
    char linha_v_objetivo[4096];
    fscanf(arq_v_objetivos, "%s", linha_v_objetivo);

    /* Alocando o vetor para armazenar o vetor de objetivo "i-ésimo". */
    float v_objetivo[qtd_itens_v_objetivo];

    /* Extraindo o primeiro item do vetor de objetivo. */
    v_objetivo[0] = atof(strtok(linha_v_objetivo, ";\n\0"));

    /* Extraindo os demais itens do vetor de objetivo. */
    for (int j = 1; j < qtd_itens_v_objetivo; j++)
    {
      /* Extraindo o item "j-ésimo" do vetor de objetivo. */
      v_objetivo[j] = atof(strtok(NULL, ";\n\0"));
    }

    /* Copiando o vetor no hospedeiro do vetor objetivo para a memória do
     * adaptador gráfico.
     */
    float * d_alvo;
    cudaMalloc((void **) &d_alvo, sizeof(float) * qtd_itens_v_objetivo);
    cudaMemcpy(d_alvo, v_objetivo, sizeof(float) * qtd_itens_v_objetivo,
	       cudaMemcpyHostToDevice);

    /* Por fim, colocando no padrão o vetor de objetivo extraido do
     * arquivo. */
    padroes[i].d_alvo = d_alvo;
  }

  fclose(arq_amostras);
  fclose(arq_v_objetivos);

  return padroes;
}

float calcular_taxa_acerto_PerceptronMulticamdas(PerceptronMulticamadas * pm,
						 PadraoTreinamento * padroes_teste,
						 int qtd_padroes_teste)
{

  /* Alocando na memória do adaptador gráfico a variável que irá armazenar
   * o erro dos padrões calculado para cada iteração.
   */
  float * d_erro_padrao;
  cudaMalloc((void **) &d_erro_padrao, sizeof(float));

  /* Percorrendo os padrões de teste e calculando o erro global. */
  float h_erro_global = 0;
  float h_erro_padrao;
    
  for (int i = 0; i < qtd_padroes_teste; i++)
  {
    /* Alimentando a rede com o padrão de teste "i-ésimo". */
    alimentar_PerceptronMulticamadas(pm, padroes_teste[i].d_amostra);

    /* Calculando o erro dos neurônios da última camada. */
    funcao_calcular_erro_rprop_neuronio_ultima_camada_sequencial_kernel<<<1, 1>>> // <- Kernel sequencial
      (*pm->v_cam[pm->qtd_camadas - 1], padroes_teste[i].d_alvo, d_erro_padrao);

    /* Copiando o erro do padrão armazenado no adaptador gráfico para o hospedeiro. */
    cudaMemcpy(&h_erro_padrao, d_erro_padrao, sizeof(float),
	     cudaMemcpyDeviceToHost);

    h_erro_global += h_erro_padrao;
  }

  /* Desalocando variáveis que não são mais necessárias da memória
   * do adaptador gráfico.
   */
  cudaFree(d_erro_padrao);

  /* Retornando o erro MSE calculado. */
  return h_erro_global / qtd_padroes_teste;
}

HistoricoTreinamento *
HistoricoTreinamento_inicializar(PerceptronMulticamadas * pm,
				 float taxaAprendizagem,
				 float erroDesejado)
{
  /* Alocando a estrutura do histórico na memória. */
  HistoricoTreinamento * historicoTreinamento;
  historicoTreinamento =
    (HistoricoTreinamento *) malloc(sizeof(HistoricoTreinamento));


  /* Populando a variável "strConfigRedeNeural". */
  historicoTreinamento->strConfigRedeNeural[0] = '\0';
  char strQtdNeuroniosCamada[10];
  sprintf((char *) &strQtdNeuroniosCamada, "%d", pm->qtd_neuronios_entrada);
  strcat(historicoTreinamento->strConfigRedeNeural, (char *) &strQtdNeuroniosCamada);
  
  for (int i = 0; i < pm->qtd_camadas; i++)
  {
    sprintf((char *) &strQtdNeuroniosCamada, "-%d", pm->v_cam[i]->qtd_neuronios);
    strcat(historicoTreinamento->strConfigRedeNeural, (char *)&strQtdNeuroniosCamada);
  }

  /* Preenchendo os atributos. */
  historicoTreinamento->qtdNeuroniosEntrada = pm->qtd_neuronios_entrada;
  historicoTreinamento->qtdCamadas = pm->qtd_camadas;
  historicoTreinamento->taxaAprendizagem = taxaAprendizagem;
  historicoTreinamento->erroDesejado = erroDesejado;
  historicoTreinamento->listaEpocas = NULL;

  /* Retornando a estrutura criada. */
  return historicoTreinamento;
}

void HistoricoTreinamento_adicionarInfoEpoca(HistoricoTreinamento *
					           historicoTreinamento,
					           float duracaoSegs,
					           float erroGlobal)
{
  /* Alocando o nó para ser inserido na lista. */
  struct ListaNo * no = (struct ListaNo *) malloc(sizeof(struct ListaNo));
  no->dado.duracaoSegs = duracaoSegs;
  no->dado.erroGlobal = erroGlobal;
  no->proxNo = NULL;
  
  /* Verificando se não há nenhuma época inserida ainda
  na lista. */
  if (historicoTreinamento->listaEpocas == NULL)
  {
    historicoTreinamento->listaEpocas = no;
  }
  else
  {
    /* Localizando o último nó da lista. */
    struct ListaNo * noAtual;
    noAtual = historicoTreinamento->listaEpocas;
    while (noAtual->proxNo != NULL)
    {
      noAtual = noAtual->proxNo;
    }
      
    /* Por fim inserindo o último nó na "última posição" 
    da lista. */
    noAtual->proxNo = no;
  }
}

void HistoricoTreinamento_gerarArquivoCSV(HistoricoTreinamento * 
					  historicoTreinamento,
					  char * nomeArquivo)
{
  /* Tentando criar o arquivo para gravação. */
  FILE * arquivoCSV;
  arquivoCSV = fopen(nomeArquivo, "a");

  /* Verificando se o arquivo foi aberto com sucesso. */
  if (arquivoCSV == NULL)
    return; // Não conseguiu abrir arquivo.

  /* Gravando a primeira linha do arquivo que irá conter a
  a arquitetura da rede, a quantidade de padrões de treinamento,
  taxa de aprendizagem e o erro desejado respectivamente. */

  /* Imprimindo a arquitetura da rede. */
  fprintf(arquivoCSV, "%s%c", historicoTreinamento->strConfigRedeNeural,
	  DELIMITADOR_CSV);

  /* Imprimindo a taxa de aprendizagem e o erro desejado. */
  fprintf(arquivoCSV, "%f%c", historicoTreinamento->taxaAprendizagem, DELIMITADOR_CSV);
  fprintf(arquivoCSV, "%f%c\n", historicoTreinamento->erroDesejado, DELIMITADOR_CSV);

  /* Imprimindo as informações sobre as épocas. */
  struct ListaNo * noAtual = historicoTreinamento->listaEpocas;
  int numEpoca = 0;
  while (noAtual != NULL)
  {
    /* Imprimindo o número da época, duração da época em segundos
    e por fim o erro global após o treinamento da mesma. */
    fprintf(arquivoCSV, "%d%c", ++numEpoca, DELIMITADOR_CSV);
    fprintf(arquivoCSV, "%f%c", noAtual->dado.duracaoSegs, DELIMITADOR_CSV);
    fprintf(arquivoCSV, "%f%c\n", noAtual->dado.erroGlobal, DELIMITADOR_CSV);

    /* Indo para a próxima época. */
    noAtual = noAtual->proxNo;
  }

  /* Fechando o arquivo... */
  fclose(arquivoCSV);
}
