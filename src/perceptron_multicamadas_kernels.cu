#include "perceptron_multicamadas_kernels.cuh"

__device__ float funcao_degrau_dev(float z)
{
  return (z >= 0) ? 1 : 0;
}

__device__ float derivada_funcao_degrau_dev(float val_degrau)
{
  return 1.0;
}

__device__ float funcao_sigmoide_dev(float z)
{
  return 1.0 / (1.0 + expf(-z));
}

__device__ float derivada_funcao_sigmoide_dev(float res_calc_sigmoide)
{
  return res_calc_sigmoide * (1.0 - res_calc_sigmoide);
}

__device__ float funcao_tang_hiperbolica_dev(float z)
{
  return tanhf(z);
}

__device__ float derivada_funcao_tang_hiperbolica_dev(float val_tang_hiperbolica)
{
  return 1 - (val_tang_hiperbolica * val_tang_hiperbolica);
}

__global__ void funcao_ativacao_neuronio_primeira_camada_kernel
(const Camada c, const float * d_amostra, int qtd_neuronios_entrada)
{
  /* Calculando qual neurônio a thread atual irá calcular
   * a função de integração.
   */
  unsigned int neuronio = blockIdx.x * blockDim.x + threadIdx.x;
  
  /* Verificando se o índice gerado acima é válido (não ultrapassa
   * a quantidade de neurônios da camada).
   */
  if (neuronio >= c.qtd_neuronios)
  {
    /* Caso positivo, a thread é encerrada. */
    return;
  }

  /* Variável que irá armazenar o valor da função de integração
   * para este neurônio.
   */
  float val_func_int = 0.0;
  
  /* Variável que irá referênciar os pesos do neurônio respectivo. */
  float * w = &c.d_W[qtd_neuronios_entrada * neuronio];
  
  /* Calculando o valor da função de integração do neurônio... */
  for (int i = 0; i < qtd_neuronios_entrada; i++)
  {
    /* Somando a amostra na posição "i-ésima" pelo respectivo peso. */
    val_func_int += w[i] * d_amostra[i];
  }

  /* Por fim calculando a ativação do neurônio (usando o bias) junto com
   * sua derivada. */
  float ativacao_neuronio;

  switch (c.funcao_ativacao)
  {
  case Identidade:
    c.d_n_ativacao[neuronio] = val_func_int + c.d_v_bias[neuronio];
    c.d_n_derivada[neuronio] = 1;
    break;
  case Degrau:
    ativacao_neuronio = funcao_degrau_dev(val_func_int +
					  c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_degrau_dev(ativacao_neuronio);
    break;
  case Sigmoide:
    ativacao_neuronio = funcao_sigmoide_dev(val_func_int +
					    c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_sigmoide_dev(ativacao_neuronio);
    break;
  case Tang_Hiperbolica:
    ativacao_neuronio = funcao_tang_hiperbolica_dev(val_func_int +
						    c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_tang_hiperbolica_dev(ativacao_neuronio);
  }
}

__global__ void funcao_ativacao_neuronio_camada_kernel(const Camada c_ant, const Camada c)
{
  /* Calculando qual neurônio a thread atual irá calcular
   * a função de integração.
   */
  unsigned int neuronio = blockIdx.x * blockDim.x + threadIdx.x;
  
  /* Verificando se o índice gerado acima é valido (não ultrapassa
   * a quantidade de neurônios da camada).
   */
  if (neuronio >= c.qtd_neuronios)
  {
    /* Caso positivo, a thread é encerrada. */
    return;
  }

  /* Variável que irá armazenar o valor da função de integração
   * para este neurônio.
   */
  float val_func_int = 0.0;
  
  /* Variável que irá referênciar os pesos do neurônio respectivo. */
  float * w = &c.d_W[c_ant.qtd_neuronios * neuronio];

  /* Calculando o valor da função de integração do neurônio... */
  for (int i = 0; i < c_ant.qtd_neuronios; i++)
  {
    /* Somando a ativação do neurônio da camada anterior na posição
     * "i-ésima" pelo respectivo peso. */
    val_func_int += w[i] * c_ant.d_n_ativacao[i];
  }
  
  /* Por fim calculando a ativação do neurônio (usando o bias) junto com
   * sua derivada. */
  float ativacao_neuronio;

  switch (c.funcao_ativacao)
  {
  case Identidade:
    c.d_n_ativacao[neuronio] = val_func_int + c.d_v_bias[neuronio];
    c.d_n_derivada[neuronio] = 1;
    break;
  case Degrau:
    ativacao_neuronio = funcao_degrau_dev(val_func_int +
					  c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_degrau_dev(ativacao_neuronio);
    break;
  case Sigmoide:
    ativacao_neuronio = funcao_sigmoide_dev(val_func_int +
					    c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_sigmoide_dev(ativacao_neuronio);
    break;
  case Tang_Hiperbolica:
    ativacao_neuronio = funcao_tang_hiperbolica_dev(val_func_int +
						    c.d_v_bias[neuronio]);
    c.d_n_ativacao[neuronio] = ativacao_neuronio;
    c.d_n_derivada[neuronio] = derivada_funcao_tang_hiperbolica_dev(ativacao_neuronio);
  }
}

__global__ void funcao_atualizacao_pesos_neuronio_primeira_camada_kernel
(const Camada c, const float * d_amostra, float taxa_aprendizagem, int qtd_neuronios_entrada)
{
  /* Calculando qual neurônio a thread atual irá atualizar os pesos. */
  unsigned int neuronio = blockIdx.x * blockDim.x + threadIdx.x;
  
  /* Verificando se o índice gerado acima é valido (não ultrapassa
   * a quantidade de neurônios da camada).
   */
  if (neuronio >= c.qtd_neuronios)
  {
    /* Caso positivo, a thread é encerrada. */
    return;
  }

  /* Variável que irá referênciar os pesos do neurônio respectivo. */
  float * w = &c.d_W[qtd_neuronios_entrada * neuronio];
  
  /* Variável que irá armazenar o erro retropropagado no neurônio
   * respectivo.
   */
  float n_erro_rprop = c.d_n_erro_rprop[neuronio];

  /* Percorrendo os pesos e atualizando os mesmos. */
  for (int i = 0;i < qtd_neuronios_entrada; i++)
  {
    /* Atualizand o peso "i-ésimo" do neurônio. */
    w[i] += -taxa_aprendizagem * d_amostra[i] * n_erro_rprop;
  }
  
  /* Atualizando o bias do neurônio. */
  c.d_v_bias[neuronio] += -taxa_aprendizagem * n_erro_rprop;
}

__global__ void funcao_atualizacao_pesos_neuronio_camada_kernel
(const Camada c_ant, const Camada c, float taxa_aprendizagem)
{
  /* Calculando qual neurônio a thread atual irá atualizar os pesos. */
  unsigned int neuronio = blockIdx.x * blockDim.x + threadIdx.x;

  /* Verificando se o índice gerado acima é valido (não ultrapassa
   * a quantidade de neurônios da camada).
   */
  if (neuronio >= c.qtd_neuronios)
  {
    /* Caso positivo, a thread é encerrada. */
    return;
  }

  /* Variável que irá referênciar os pesos do neurônio respectivo. */
  float * w = &c.d_W[c_ant.qtd_neuronios * neuronio];

  /* Variável que irá armazenar o erro retropropagado no neurônio
   * respectivo.
   */
  float n_erro_rprop = c.d_n_erro_rprop[neuronio];

  /* Percorrendo os pesos e atualizando os mesmos. */
  for (int i = 0; i < c_ant.qtd_neuronios; i++)
  {
    /* Atualizand o peso "i-ésimo" do neurônio. */
    w[i] += -taxa_aprendizagem * c_ant.d_n_ativacao[i] * n_erro_rprop;
  }
  
  /* Atualizando o bias do neurônio. */
  c.d_v_bias[neuronio] += -taxa_aprendizagem * n_erro_rprop;
}

__global__ void funcao_calcular_erro_rprop_neuronio_ultima_camada_sequencial_kernel
(const Camada c, const float * d_alvo, float * d_erro_padrao)
{

  /* Variável que irá armazenar o valor do erro padrão. */
  float erro_padrao = 0.0;

  /* Percorrendo todos os neurônios da camada. */
  for (int neuronio = 0; neuronio < c.qtd_neuronios; neuronio++)
  {
    /* Calculando o erro da saída deste neurônio. */
    float erro_neuronio = c.d_n_ativacao[neuronio] - d_alvo[neuronio];
    
    /* Calculando o erro retropropagado e inserindo o mesmo na respectiva
     * estrutura. */
    c.d_n_erro_rprop[neuronio] = erro_neuronio * c.d_n_derivada[neuronio];
    
    /* Somando o erro calculado para o neurônio elevado a dois na
     * variável do erro do padrão apresentado à rede.
     */
    erro_padrao += 0.5 * powf(erro_neuronio, 2);
  }

  /* "Retornando" o erro calculado para o padrão. */
  *d_erro_padrao = erro_padrao;
}

__global__ void funcao_calcular_erro_rprop_neuronio_camada_kernel(const Camada c, const Camada c_pos)
{
  /* Calculando qual neurônio a thread atual irá calcular
   * o erro retropropagado.
   */
  unsigned int neuronio = blockIdx.x * blockDim.x + threadIdx.x;

  /* Verificando se o índice gerado acima é valido (não ultrapassa
   * a quantidade de neurônios da camada).
   */
  if (neuronio >= c.qtd_neuronios)
  {
    /* Caso positivo, a thread é encerrada. */
    return;
  }

  /* Calculando a soma dos erros da camada posterior. */
  float soma_erro = 0.0;
  
  for (int i = 0; i < c_pos.qtd_neuronios; i++)
  {
    /* Coletando o peso do neurônio "i-ésimo" da camada posterior
     * que se conecta ao respectivo neurônio "n-ésimo" que está tendo seu
     * erro calculado. */
    float w = c_pos.d_W[c.qtd_neuronios * i + neuronio];
      
    /* Calculando o erro do neurônio "i-ésimo" da camada posterior
     * multiplicado pelo respectivo peso da camada posterior que se conecta
     * a este neurônio e somando.
     */
    soma_erro += w * c_pos.d_n_erro_rprop[i];
  }

  /* Por fim, multiplicando o erro somado pela derivada
   * do neurônio da camada atual.
   */
  c.d_n_erro_rprop[neuronio] = c.d_n_derivada[neuronio] * soma_erro;
}
