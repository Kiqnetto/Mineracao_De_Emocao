import nltk
import data
from nltk.metrics import ConfusionMatrix
#Coloca na variável o conjunto de stopwords já definidas pela biblioteca
stop_Words = nltk.corpus.stopwords.words('portuguese')

#Faz o stemming, que é extrair os sufixos das palavras
def aplica_Stemming(texto):
    stemmer = nltk.stem.RSLPStemmer()
    novo_Texto = []
    for (palavras, emocao) in texto:
        radical_Palavras = [str(stemmer.stem(p)) for p in palavras.split() if p not in stop_Words]
        novo_Texto.append((radical_Palavras, emocao))
    return novo_Texto

stemming_treinamento = aplica_Stemming(data.base_treinamento)
#print(stemming_treinamento)

# -- Seleciona as palavras que serão nossos parâmetros --
def seleciona_Palavras(texto):
    todas_Palavras = []
    for (palavras, emocao) in texto:
        todas_Palavras.extend(palavras)
    return todas_Palavras

parametros_treinamento = seleciona_Palavras(stemming_treinamento)
#print(parametros_treinamento)

# -- Verificamos a frequência em que as palavras (parâmetros) aparecem --
def frequencia_palavras(palavras):
    palavras = nltk.FreqDist(palavras)
    return palavras

frequencia_treinamento = frequencia_palavras(parametros_treinamento)

# -- Tira as palavras repetidas --
def palavras_sem_repeticao(frequencia):
    freq = frequencia.keys()
    return freq

palavras_unicas_treinamento = palavras_sem_repeticao(frequencia_treinamento)
#print(palavras_unicas_treinamento)

# -- Seleciona cada palavra de um texto --
def seleciona_palavra_frase(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavras_unicas_treinamento:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

# -- Monta a tabela em que verifica se cada frase possui a palavra da tabela --
base_compl_treinamento = nltk.classify.apply_features(seleciona_palavra_frase,stemming_treinamento)

# -- Constrói a tabela de probabilidades --
classificador = nltk.NaiveBayesClassifier.train(base_compl_treinamento)

#Mostra passado um parâmetro inteiro, a probabilidade de tal palavra em relação a dois classificadores
#print(classificador.show_most_informative_features(10))

# -- Faz os processos anteriores com a base de teste --
# -- Aplica o procedimento de deixar apenas o radical das palavras --
stemming_teste = aplica_Stemming(data.base_teste)

# -- Seleciona as palavras --
parametros_teste = seleciona_Palavras(stemming_teste)

# -- Freq. que as palavras aparecem --
frequencia_teste = frequencia_palavras(parametros_teste)

# -- Seleciona as palavras únicas --
palavras_unicas_teste = palavras_sem_repeticao(frequencia_teste)

# -- Monta a base completa de teste após o pré-processamento --
base_compl_teste = nltk.classify.apply_features(seleciona_palavra_frase, stemming_teste)

classificador_teste = nltk.classify.accuracy(classificador, base_compl_teste)
print('Acurácia: ' + str(classificador_teste))

# -- Avaliação dos resultados --
erros = []
for (frase, classe) in base_compl_teste:
    resultado = classificador.classify(frase) # previsão do algoritmo
    if resultado != classe:
        erros.append((classe, resultado, frase))

# -- Visualização dos resultados --
resultado_esperado = []
resultado_previsto = []
for (frase, classe) in base_compl_teste:
    resultado = classificador.classify(frase)
    resultado_previsto.append(resultado)
    resultado_esperado.append(classe)

matriz = ConfusionMatrix(resultado_esperado, resultado_previsto)
print(matriz)

# -- P/ Teste --
teste = 'estou muito alegre hoje'
teste_stemming = []
stemmer = nltk.stem.RSLPStemmer()
for (palavras_treinamento) in teste.split():
    com_stem = [p for p in palavras_treinamento.split()]
    teste_stemming.append(str(stemmer.stem(com_stem[0])))

novo = seleciona_palavra_frase(teste_stemming)
print(teste_stemming)
print(teste)
distribuicao = classificador.prob_classify(novo)
for classe in distribuicao.samples():
    print("%s: %f" % (classe, distribuicao.prob(classe)))







