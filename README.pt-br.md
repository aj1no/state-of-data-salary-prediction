# Predição de Faixa Salarial de Profissionais de Dados no Brasil com Machine Learning

*[Read in English](README.md)*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/aj1no/state-of-data-salary-prediction/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/aj1no/state-of-data-salary-prediction/actions)

* **Instituição:** Faculdade de Tecnologia de Jundiaí (FATEC Jundiaí)  
* **Disciplina:** Inteligência Computacional  
* **Professor:** Me. Mateus Guilherme Fuini  
* **Autores:**
    * Rodolfo Vinicius Cima Takemoto
    * Tiago Galhardo Avelar

---

## 1. Descrição do Problema

No cenário corporativo contemporâneo, a área de Tecnologia da Informação — com ênfase em Ciência de Dados, Engenharia de Dados e Business Intelligence — experimentou uma expansão sem precedentes. No entanto, a precificação do trabalho e a definição de cargos e salários no Brasil enfrentam distorções devido a variações geográficas, modalidades de contratação (CLT vs. PJ), porte das organizações e diversidade de stacks tecnológicas exigidas pelas empresas. 

Este projeto aborda o problema de prever a faixa salarial de profissionais de dados no Brasil sob a ótica do Aprendizado de Máquina Supervisionado (Classificação). Compreender quais fatores estruturais da carreira (como senioridade, tempo de experiência e número de tecnologias dominadas) mais influenciam a faixa salarial permite que profissionais orientem melhor suas carreiras e que departamentos de Recursos Humanos calibrem suas propostas comerciais de contratação.

---

## 2. Descrição do Dataset

O projeto utiliza o conjunto de dados **State of Data Brazil 2024-2025**, a mais completa pesquisa sobre o mercado de trabalho em dados do país, promovida pela comunidade Data Hackers em parceria com a consultoria Bain & Company. 
* **Volume de Dados:** O arquivo bruto possui originalmente 5.217 respondentes e 403 variáveis/colunas.
* **Conteúdo:** O levantamento engloba informações demográficas (gênero, idade, localização), formação acadêmica (nível de ensino, curso), situação profissional (cargo, senioridade, modelo de trabalho, setor e porte da empresa) e perguntas booleanas sobre o uso cotidiano de tecnologias específicas (bancos de dados, linguagens de programação e ferramentas de BI).
* **Arquivo Base:** O dataset é carregado a partir do CSV extraído e armazenado em `data/raw/`.

---

## 3. Objetivo

O objetivo geral deste projeto é construir, otimizar e avaliar um pipeline completo de Machine Learning supervisionado para classificar a faixa salarial de profissionais da área de dados no Brasil em três categorias: Baixa faixa salarial, Média faixa salarial e Alta faixa salarial.

---

## 4. Metodologia

O desenvolvimento seguiu a abordagem clássica de projetos de Ciência de Dados:
1. **Seleção Seletiva de Recursos (Feature Selection):** Devido ao alto número de colunas, filtramos e mantivemos as variáveis mais explicativas e estruturais da carreira, reduzindo as colunas preditoras a um conjunto de 12 colunas centrais.
2. **Tratamento de Codificação e Nulos:** As colunas de texto foram limpas de artefatos de codificação (como acentos corrompidos no carregamento) para assegurar o funcionamento dos mapeamentos categóricos. Registros com a variável-alvo ausente foram descartados.
3. **Engenharia de Atributos:** Criação de novas variáveis derivadas para enriquecer o contexto fornecido ao modelo (incluindo `Trabalha_Remoto`, `Experiencia_Categoria` e `Perfil_Tecnico_Qtd`).
4. **Análise de Outliers e Visualizações:** Identificação exploratória de outliers pelo método IQR e geração de visualizações descritivas (histograma da stack, countplots de senioridade e experiência, boxplot da quantidade de tecnologias por faixa e heatmap exploratório de correlação de Pearson com variáveis ordinais/binárias ordinalizadas temporariamente).
5. **Particionamento Estruturado:** Divisão treino/teste estratificada para manter o balanceamento das classes.
6. **Pipelines de Pré-processamento e Capping:** Uso de `ColumnTransformer` e `Pipeline` para aplicar imputação, tratamento robusto de outliers por capping (através do custom transformer `IQRCapper`) e redimensionamento numérico (`StandardScaler`), além da codificação de variáveis categóricas (`OneHotEncoder`). Todo o pré-processamento é encapsulado no pipeline para evitar data leakage.
7. **Otimização e Validação:** Ajuste de hiperparâmetros de um classificador de Regressão Logística via `GridSearchCV` (testando diferentes valores de regularização `C` e solvers como `lbfgs` e `newton-cg` com 1000 iterações máximas e `error_score="raise"`) associado a um esquema de validação cruzada `StratifiedKFold`.

---

## 5. Mapeamento da Variável-Alvo e Data Leakage

A coluna original `2.h_faixa_salarial` possui 13 faixas salariais declaradas. Para viabilizar uma classificação robusta e balanceada, criamos uma variável-alvo simplificada mapeada da seguinte forma:

| Faixa Original de Renda Mensal | Classe Alvo Simplificada |
| :--- | :--- |
| *Menos de R$ 1.000/mês* até *de R$ 4.001/mês a R$ 6.000/mês* | **Baixa faixa salarial** (até R$ 6.000) |
| *de R$ 6.001/mês a R$ 8.000/mês* até *de R$ 8.001/mês a R$ 12.000/mês* | **Média faixa salarial** (de R$ 6.001 a R$ 12.000) |
| *de R$ 12.001/mês a R$ 16.000/mês* até *Acima de R$ 40.001/mês* | **Alta faixa salarial** (acima de R$ 12.000) |

### Prevenção de Data Leakage
Para evitar o vazamento de informações que causam falsos otimismos na acurácia do modelo:
1. **Descarte de Vazadores Diretos:** Removemos colunas de remuneração bruta secundária, bônus adicionais, benefícios e pretensão salarial antes da modelagem.
2. **Encapsulamento de Transformadores:** O cálculo do redimensionamento numérico (`StandardScaler`) e a codificação categórica (`OneHotEncoder`) foram encapsulados no `Pipeline`. O ajuste (`fit`) é restrito aos dados de treino, e o conjunto de teste é tratado estritamente de forma cega com transformações de inferência (`transform`).

---

## 6. Engenharia de Atributos

Criamos três variáveis derivadas para o treinamento do classificador:

1. **`Experiencia_Categoria`:** Agrupamento lógico dos anos de atuação declarados na área de dados em três níveis (`Até 2 anos`, `3 a 6 anos`, `Mais de 6 anos`).
2. **`Trabalha_Remoto`:** Atributo binário (`Sim`/`Não`) derivado do modelo de trabalho atual. Indica se o profissional trabalha de forma 100% remota.
3. **`Perfil_Tecnico_Qtd`:** Soma das marcações booleanas de tecnologias utilizadas diariamente (seção 4 da pesquisa). É uma variável contínua que mede a amplitude de ferramentas dominadas pelo profissional.

---

## 7. Resultados do Modelo

O modelo selecionado foi a **Regressão Logística** otimizada por busca em grade.

* **Melhores Hiperparâmetros encontrados:** `{'model__C': 1.0, 'model__solver': 'lbfgs', 'model__max_iter': 1000}`
* **Acurácia Média na Validação Cruzada (5-Fold):** 73,47% (Desvio Padrão: 1,69%)
* **Acurácia Final no Conjunto de Teste:** 72,46%

### Relatório de Classificação Final (Teste)

| Classe Salarial | Precisão (Precision) | Revocação (Recall) | F1-Score | Suporte (Instâncias) |
| :--- | :---: | :---: | :---: | :---: |
| **Alta faixa salarial** | 0.76 | 0.82 | 0.79 | 368 |
| **Baixa faixa salarial** | 0.81 | 0.73 | 0.77 | 258 |
| **Média faixa salarial** | 0.63 | 0.62 | 0.62 | 347 |
| *Acurácia Geral* | | | **0.72** | **973** |

### Matriz de Confusão

```
                     Classe Predita
                    Baixa   Média   Alta
Classe   Baixa   [   189,     63,     6 ]
Real     Média   [    43,    216,    88 ]
         Alta    [     2,     66,   300 ]
```

---

## 8. Conclusões e Limitações

### Conclusões Acadêmicas
* O classificador apresentou desempenho sólido com acurácia de 72,46% em dados de teste.
* As faixas extremas (Alta e Baixa) obtiveram os melhores índices de classificação (F1-score de 79% e 77% respectivamente), demonstrando que profissionais no início de carreira e profissionais sêniores/gestores possuem perfis e stacks técnicas bem distintos.
* A faixa Média salarial demonstrou ser a mais complexa de classificar (F1-score de 62%), o que é condizente com o mercado brasileiro, onde profissionais plenos ou seniores de pequenas empresas muitas vezes possuem remunerações próximas a profissionais juniores de grandes multinacionais.

### Limitações do Estudo
* O dataset baseia-se em respostas voluntárias declaradas, o que pode apresentar vieses de autopreenchimento e amostragem.
* A regressão logística impõe relações lineares e aditivas sobre os coeficientes, o que impede a representação direta de interações de segundo grau complexas (por exemplo, o impacto combinado de morar no Sudeste E dominar computação em nuvem).

### Melhorias Futuras
1. Experimentar modelos baseados em árvores (ex: Random Forest, Gradient Boosting ou XGBoost) que capturam relações não lineares e interações nativamente.
2. Atribuir pesos qualitativos às tecnologias em vez de apenas somá-las (ex: dominar Spark/Kubernetes possui maior impacto salarial do que dominar apenas Excel).
3. Avaliar técnicas de balanceamento de classes se a variância dos dados aumentar em edições futuras.

---

## Execução no Google Colab com KaggleHub

O projeto foi adaptado para permitir a execução automatizada tanto em ambientes locais quanto no Google Colab. A importação dos dados ocorre de forma robusta e dinâmica seguindo o fluxo abaixo:

1. **Download Automático:** O notebook tenta baixar o conjunto de dados diretamente do Kaggle utilizando o slug oficial `datahackers/state-of-data-brazil-20242025` através da biblioteca `kagglehub`.
2. **Autenticação:** Caso o Kaggle solicite autenticação, o usuário pode configurar suas credenciais (token de API do Kaggle). Caso contrário, a biblioteca fará o download de forma anônima se permitido, ou apresentará um erro controlado.
3. **Fallback para Upload Manual:** Se o download automatizado pelo `kagglehub` falhar por qualquer motivo (falta de conectividade, ausência de credenciais ou pendências nos termos do dataset), o notebook aciona uma rotina secundária de upload manual no Google Colab via `files.upload()`. O usuário precisará fazer o envio do arquivo CSV correspondente à pesquisa.
4. **Armazenamento no GitHub:** Graças à carga dinâmica por API ou upload sob demanda, o arquivo CSV bruto do dataset não precisa (e não deve) ser armazenado no repositório do GitHub.
