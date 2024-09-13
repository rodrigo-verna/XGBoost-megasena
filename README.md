# XGBoost-megasena
# Previsão de Números da Mega-Sena Usando XGBoost

## Visão Geral

Este modelo, para fins de estudos, tem como objetivo prever os números com alguma chance de serem sorteados em futuros concursos da Mega-Sena, utilizando o algoritmo de machine learning **XGBoost**. O modelo é treinado com dados históricos de concursos anteriores, aproveitando uma variedade de features, como a frequência dos números, distribuição entre pares e ímpares, agrupamento por regiões, entre outras métricas derivadas. O projeto inclui código para treinar o modelo, avaliar seu desempenho e fornecer recomendações dos 20 números com maior probabilidade de serem sorteados nos próximos concursos.

**NÃO SE ENGANE!** A aletoriedade da Mega-Sena e das loterias em geral impede qualquer modelo matemático prever números com exatidão. 

## Funcionalidades

- **Pré-processamento de Dados**: Os dados históricos são limpos e enriquecidos com novas features, como o número de pares/ímpares sorteados, desvio padrão e agrupamentos de números por regiões.
- **Treinamento do Modelo com XGBoost**: O modelo usa o XGBoost para classificação, prevendo se cada número de 1 a 60 será sorteado ou não em futuros concursos.
- **Métricas de Desempenho**: Após o treinamento, o modelo é avaliado utilizando métricas como **precisão**, **revocação**, **F1-score** e a **matriz de confusão**, além de contabilizar os acertos e erros das previsões.
- **Ajuste de Limiar de Decisão**: O limiar de decisão é ajustado para tentar melhorar a sensibilidade na previsão dos números sorteados, equilibrando entre precisão e revocação.
- **Sistema de Recomendação**: O modelo fornece recomendações dos 20 números mais prováveis de serem sorteados, com base nas previsões feitas a partir dos dados históricos.

## Arquivos

- **`XGBoost-megasena.py`**: Script principal para treinar o modelo XGBoost, avaliar seu desempenho e gerar as recomendações finais.
- **`data/`**: Diretório que contém os dados históricos da Mega-Sena utilizados para treinar o modelo.
- **`README.md`**: Este arquivo, descrevendo o projeto e como utilizá-lo.

## Como Utilizar

1. Clone este repositório:
    ```bash
    git clone https://github.com/rodrigo-verna/XGBoost-megasena.git
    ```
2. Navegue até a pasta do projeto:
    ```bash
    cd XGBoost-megasena
    ```
3. Execute o script de treinamento do modelo:
    ```bash
    python XGBoost-megasena.py
    ```
4. Após o treinamento, o modelo irá exibir:
    - As métricas de desempenho, incluindo acurácia, precisão, revocação e F1-score.
    - Os 20 números com maior probabilidade de serem sorteados nos próximos concursos.

## Métricas de Desempenho

O modelo é avaliado com as seguintes métricas:

- **Acurácia**: Medida geral da eficácia do modelo.
- **Precisão**: Proporção de números previstos como sorteados que realmente foram sorteados.
- **Revocação**: Capacidade do modelo de identificar corretamente os números que foram sorteados.
- **F1-Score**: Combinação entre precisão e revocação.
- **Matriz de Confusão**: Mostra detalhadamente os acertos e erros do modelo (verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos).

## Limitações

Embora o modelo seja projetado para identificar padrões nos dados históricos da Mega-Sena, é importante notar que os sorteios de loteria são, por natureza, aleatórios. As previsões do modelo são baseadas em padrões observados no passado, portanto, não haverá sucesso. Os fins desse projeto são apenas de ensino e aprendizado de como implementar esse algoritmo. 
