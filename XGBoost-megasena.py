import pandas as pd
import random
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1 - Carregando dados históricos 
df = pd.read_csv('CAMINHO DO SEU DIRETÓRIO', sep=';')

# 2 - Tratamento dos dados e criação de features 
# Selecionar apenas as colunas relevantes para o modelo: Data do Sorteio e as bolas sorteadas
dfLimpo = df[['Data do Sorteio', 'Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']]

# Converter a coluna "Data do Sorteio" para o formato de data apropriado
dfLimpo.loc[:, 'Data do Sorteio'] = pd.to_datetime(dfLimpo['Data do Sorteio'], format='%d/%m/%Y')

# Verificar se há duplicatas e removê-las (se necessário)
dfRelevante = dfLimpo.drop_duplicates()

# Definir as colunas com os números sorteados
numerosSorteados = ['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']

# Converter a coluna 'Data do Sorteio' para o formato datetime
dfRelevante['Data do Sorteio'] = pd.to_datetime(dfRelevante['Data do Sorteio'], format='%d/%m/%Y')

# Concatenar as colunas de números sorteados em uma única série para contar a frequência de cada número
numeros = pd.concat([dfRelevante[col] for col in numerosSorteados])

# # Contar a frequência de cada número sorteado e garantir que o tipo seja inteiro
# frequenciaNumeros = numeros.value_counts().astype(int)  # Converter a frequência para tipo inteiro

# # Criar colunas separadas para exibir a frequência de cada número individualmente e garantir que os valores sejam inteiros
# for col in numerosSorteados:
#     dfRelevante[f'freqResult{col}'] = dfRelevante[col].map(frequenciaNumeros).astype(int)

# Verificar as primeiras linhas para garantir que as frequências foram calculadas corretamente
#print(dfRelevante[[f'freqResult{col}' for col in numerosSorteados]].head())

# Criar as features 'numPares' e 'numIpares': número de números pares e ímpares sorteados
dfRelevante['numPares'] = dfRelevante[numerosSorteados].apply(lambda x: (x % 2 == 0).sum(), axis=1)
dfRelevante['numImpares'] = dfRelevante[numerosSorteados].apply(lambda x: (x % 2 != 0).sum(), axis=1)

# Criar a nova feature 'regioesVolante' dividindo o volante em 4 regiões
regioesVolante = {
    'Região 1': [1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 21, 22, 23, 24, 25],
    'Região 2': [6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 26, 27, 28, 29, 30],
    'Região 3': [31, 32, 33, 34, 35, 41, 42, 43, 44, 45, 51, 52, 53, 54, 55],
    'Região 4': [36, 37, 38, 39, 40, 46, 47, 48, 49, 50, 56, 57, 58, 59, 60]
}

# Função para contar quantos números sorteados estão em cada região
def contarRegiao(linha):
    regioesContagem = {regiao: 0 for regiao in regioesVolante}
    for numero in linha[numerosSorteados]:
        for regiao, numeros in regioesVolante.items():
            if numero in numeros:
                regioesContagem[regiao] += 1
    return regioesContagem

# Aplicar a função e criar novas colunas para a contagem de números por região
dfRelevante['Regiao1'] = dfRelevante.apply(lambda row: contarRegiao(row)['Região 1'], axis=1)
dfRelevante['Regiao2'] = dfRelevante.apply(lambda row: contarRegiao(row)['Região 2'], axis=1)
dfRelevante['Regiao3'] = dfRelevante.apply(lambda row: contarRegiao(row)['Região 3'], axis=1)
dfRelevante['Regiao4'] = dfRelevante.apply(lambda row: contarRegiao(row)['Região 4'], axis=1)

# Criar a feature 'desvioPadrao': desvio padrão dos números sorteados
dfRelevante['desvioPadrao'] = dfRelevante[numerosSorteados].std(axis=1)

# Criar a feature 'Media': média dos números sorteados
dfRelevante['Media'] = dfRelevante[numerosSorteados].mean(axis=1)

# Extração de dados temporais: 'diaSemana' e 'Mes' a partir da coluna 'Data do Sorteio'
dfRelevante['diaSemana'] = dfRelevante['Data do Sorteio'].dt.dayofweek  # 0 = Segunda, 6 = Domingo
dfRelevante['Mes'] = dfRelevante['Data do Sorteio'].dt.month

# 3 - Treinamento do modelo e impressão de estatísticas 

# Função para gerar uma jogada inicial aleatória de 6 números
def gerarJogada():
    return random.sample(range(1, 61), 6)

# Função para ajustar o limiar de decisão
def ajustarLimiar(probabilidades, limiar=0.5):
    return [1 if prob >= limiar else 0 for prob in probabilidades]

# Criar a contagem de acertos para cada número (de 1 a 60)
acertosTotais = {n: 0 for n in range(1, 61)}

# Variáveis para contar o número de acertos e erros totais
totalAcertos = 0
totalErros = 0
totalJogos = 0  # Para contar o total de jogos jogados

# Listas para armazenar os valores reais e previstos, para análise das métricas
valoresReais = []
valoresPrevistos = []

# Criar um dataframe simplificado com os resultados históricos (exemplo)
numerosEscolhidos = list(range(1, 61))
features = ['numPares', 'numImpares', 'Regiao1', 'Regiao2', 'Regiao3', 'Regiao4', 'desvioPadrao', 'Media', 'diaSemana', 'Mes']

# Número de iterações (ou número de concursos que o modelo vai aprender progressivamente)
maxTentativas = len(dfRelevante)  # Baseado no número de sorteios históricos

# Definir o limiar de decisão
limiar = 0.3  # Testar com 0.3, pode ajustar este valor

# Treinar o modelo XGBoost
for i in range(maxTentativas):
    # Contar o total de jogos jogados
    totalJogos += 1

    # Fazer uma jogada inicial aleatória
    jogada = gerarJogada()

    # Comparar com o resultado do sorteio correspondente (iésimo sorteio)
    resultadoReal = dfRelevante[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].iloc[i]
    acertos = set(jogada) & set(resultadoReal)

    # Atualizar a contagem de acertos para os números acertados
    if len(acertos) > 0:
        for numero in acertos:
            acertosTotais[numero] += 1
        totalAcertos += 1  # Incrementar o total de acertos
    else:
        totalErros += 1  # Incrementar o total de erros

    # Guardar os valores reais e previstos para análise posterior
    valoresReais.extend([1 if num in resultado_real.values else 0 for num in numerosEscolhidos])
    valoresPrevistos.extend([1 if num in jogada else 0 for num in numerosEscolhidos])

    # Se o modelo não acertar nada, ajustar usando o XGBoost com base nas features
    if len(acertos) == 0:
        # Definir X (features) e y (números sorteados ou não)
        X = dfRelevante[features].iloc[:i]  # Usar dados até o i-ésimo sorteio
        y = dfRelevante[['Bola1', 'Bola2', 'Bola3', 'Bola4', 'Bola5', 'Bola6']].iloc[:i].apply(lambda row: 1 if set(row) & set(jogada) else 0, axis=1)

        # Instanciar e treinar o XGBoost
        modelo = XGBClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        # Usar o modelo para prever as probabilidades para todos os números
        probas = modelo.predict_proba(dfRelevante[features].iloc[[i]])[0]

        # Ajustar o limiar de decisão para as probabilidades
        jogadaAjustada = ajustarLimiar(probas, limiar)

        # Selecionar os 6 números com as maiores probabilidades ajustadas
        jogada = [x for prob, x in zip(jogadaAjustada, numerosEscolhidos) if prob == 1][:6]

# Após todas as tentativas, calcular os números com mais acertos
recomendacoes = sorted(acertosTotais, key=acertosTotais.get, reverse=True)[:20]

# Exibir as estatísticas finais
print("Recomendação dos 20 números com maior probabilidade de serem sorteados:")
print(recomendacoes)

# Exibir a contagem de acertos e erros
print("\nEstatísticas do modelo:")
print(f"Total de acertos: {totalAcertos}")
print(f"Total de erros: {totalErros}")
print(f"Total de jogos jogados: {totalJogos}")

# Exibir os números que o modelo mais acertou e a quantidade de acertos
print("\nNúmeros que o modelo mais acertou e suas contagens:")
for numero, contagem in sorted(acertosTotais.items(), key=lambda x: x[1], reverse=True):
    if contagem > 0:  # Mostrar apenas números que o modelo acertou
        print(f"Número {numero}: {contagem} acertos")

# Relatório de classificação com métricas principais (sem o target_names)
print("\nRelatório de classificação:")
print(classification_report(valoresReais, valoresPrevistos))

# Matriz de Confusão
print("\nMatriz de Confusão:")
print(confusion_matrix(valoresReais, valoresPrevistos))
