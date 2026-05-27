# Relatório Técnico: Predição de Faixa Salarial de Profissionais de Dados no Brasil

**Projeto:** State of Data Salary Prediction  
**Instituição:** Faculdade de Tecnologia de Jundiaí (FATEC Jundiaí)  
**Disciplina:** Inteligência Computacional  
**Autores:** Rodolfo Vinicius Cima Takemoto e Tiago Galhardo Avelar  
**Base de dados:** State of Data Brazil 2024-2025  
**Data da revisão:** 27 de maio de 2026

---

## 1. Resumo Executivo

Este projeto desenvolve um pipeline de Aprendizado de Máquina supervisionado para classificar profissionais brasileiros da área de dados em três faixas salariais: **Baixa**, **Média** e **Alta**. A modelagem utiliza informações demográficas, profissionais, de senioridade, modalidade de trabalho, porte/setor da empresa e um indicador derivado da quantidade de tecnologias utilizadas pelo respondente.

O modelo final, uma **Regressão Logística** otimizada com `GridSearchCV`, alcançou **72,46% de acurácia no conjunto de teste** e **73,47% de acurácia média em validação cruzada estratificada de 5 folds**. O desempenho foi mais forte nas classes extremas, especialmente na classe de alta faixa salarial, enquanto a faixa média apresentou maior ambiguidade, comportamento esperado em um mercado com sobreposição entre experiência, cargo, região, porte da empresa e regime de contratação.

---

## 2. Contexto e Problema

A área de dados no Brasil reúne perfis profissionais muito variados, como analistas de dados, analistas de BI, engenheiros de dados, cientistas de dados, gestores e especialistas técnicos. A remuneração desses profissionais tende a variar por fatores como:

- senioridade;
- tempo de experiência;
- região de atuação;
- formação acadêmica;
- setor econômico;
- tamanho da empresa;
- modelo de trabalho;
- cargo exercido;
- amplitude da stack tecnológica.

O problema tratado é, portanto, um problema de **classificação multiclasse**, no qual o objetivo é estimar a faixa salarial de um profissional a partir de atributos observáveis da sua trajetória e do seu contexto de trabalho.

---

## 3. Dataset

O projeto utiliza a pesquisa **State of Data Brazil 2024-2025**, promovida pela comunidade Data Hackers em parceria com a Bain & Company. O arquivo bruto possui originalmente **5.217 respondentes** e mais de **400 variáveis**, abrangendo dados demográficos, acadêmicos, profissionais, salariais e tecnológicos.

Após o tratamento da variável-alvo e remoção de registros sem informação salarial, o conjunto modelado ficou com **4.863 instâncias**, divididas em:

| Partição | Instâncias |
| :--- | ---: |
| Treino | 3.890 |
| Teste | 973 |

Distribuição das classes no conjunto de teste:

| Classe | Instâncias |
| :--- | ---: |
| Alta faixa salarial | 368 |
| Média faixa salarial | 347 |
| Baixa faixa salarial | 258 |

---

## 4. Variável-Alvo

A variável original `2.h_faixa_salarial` possui 13 intervalos de renda mensal. Para reduzir esparsidade e tornar o problema mais robusto academicamente, as faixas foram agrupadas em três classes:

| Intervalo original | Classe simplificada |
| :--- | :--- |
| Até R$ 6.000/mês | Baixa faixa salarial |
| De R$ 6.001/mês a R$ 12.000/mês | Média faixa salarial |
| Acima de R$ 12.000/mês | Alta faixa salarial |

Esse agrupamento simplifica a predição, reduz o número de classes com poucos exemplos e permite avaliar perfis salariais mais amplos.

---

## 5. Seleção e Engenharia de Atributos

O dataset original possui centenas de colunas, então o projeto seleciona atributos com maior relação conceitual com remuneração e carreira. O conjunto final de preditores usado no modelo contém:

| Tipo | Atributos |
| :--- | :--- |
| Demográficos e formação | `genero`, `escolaridade`, `regiao` |
| Profissionais | `situacao_trabalho`, `setor`, `tamanho_empresa`, `gestor`, `cargo`, `nivel` |
| Derivados | `Experiencia_Categoria`, `Trabalha_Remoto`, `Perfil_Tecnico_Qtd` |

Foram criadas três variáveis derivadas:

- **`Experiencia_Categoria`**: agrupa a experiência em dados em `Até 2 anos`, `3 a 6 anos` e `Mais de 6 anos`.
- **`Trabalha_Remoto`**: indica se o respondente atua em modelo 100% remoto.
- **`Perfil_Tecnico_Qtd`**: soma as respostas booleanas de tecnologias usadas no cotidiano, servindo como proxy da amplitude técnica do profissional.

---

## 6. Metodologia de Modelagem

O fluxo de modelagem foi implementado em notebook, seguindo uma sequência típica de projeto de Ciência de Dados:

1. carregamento do dataset via KaggleHub, upload manual no Colab ou CSV local em `data/raw/`;
2. renomeação e seleção das colunas relevantes;
3. correção de encoding em campos textuais;
4. remoção de registros sem salário informado;
5. mapeamento da variável-alvo para três classes;
6. análise exploratória e geração de gráficos;
7. engenharia de atributos;
8. divisão treino/teste com estratificação;
9. construção de pipeline com pré-processamento e modelo;
10. otimização de hiperparâmetros com validação cruzada;
11. avaliação no conjunto de teste.

O pré-processamento utiliza `ColumnTransformer` e `Pipeline`, evitando vazamento de dados entre treino e teste. A variável numérica `Perfil_Tecnico_Qtd` passa por imputação de mediana, capping por IQR e padronização com `StandardScaler`. As variáveis categóricas passam por imputação de moda e codificação `OneHotEncoder(handle_unknown="ignore")`.

---

## 7. Modelo e Otimização

O classificador adotado foi a **Regressão Logística**, escolhida por ser um modelo supervisionado interpretável, eficiente e adequado como baseline forte para problemas tabulares com muitas variáveis categóricas codificadas.

Configuração de validação:

| Item | Configuração |
| :--- | :--- |
| Particionamento | 80% treino / 20% teste |
| Estratificação | Sim |
| Validação cruzada | `StratifiedKFold`, 5 folds |
| Busca de hiperparâmetros | `GridSearchCV` |
| Métrica de otimização | Acurácia |

Melhores hiperparâmetros encontrados:

```text
{'model__C': 1.0, 'model__solver': 'lbfgs', 'model__max_iter': 1000}
```

---

## 8. Resultados

| Métrica | Resultado |
| :--- | ---: |
| Acurácia média na validação cruzada | 73,47% |
| Desvio padrão da validação cruzada | 1,69% |
| Acurácia no conjunto de teste | 72,46% |

Relatório de classificação no conjunto de teste:

| Classe | Precisão | Recall | F1-score | Suporte |
| :--- | ---: | ---: | ---: | ---: |
| Alta faixa salarial | 0,76 | 0,82 | 0,79 | 368 |
| Baixa faixa salarial | 0,81 | 0,73 | 0,77 | 258 |
| Média faixa salarial | 0,63 | 0,62 | 0,62 | 347 |

A matriz de confusão mostra que o maior volume de erro está na separação entre **Média** e **Alta faixa salarial**, o que é coerente com a realidade do mercado: profissionais plenos, seniores, gestores e especialistas podem apresentar perfis próximos, mas remunerações diferentes conforme região, empresa, cargo e regime de contratação.

```text
                     Classe Predita
                    Baixa   Média   Alta
Classe   Baixa   [   189,     63,     6 ]
Real     Média   [    43,    216,    88 ]
         Alta    [     2,     66,   300 ]
```

---

## 9. Evidências Visuais

As visualizações geradas pelo notebook apoiam a interpretação dos dados e do desempenho do modelo.

![Distribuição da variável-alvo](figures/distribuicao_alvo.png)

![Senioridade versus salário](figures/senioridade_vs_salario.png)

![Experiência versus salário](figures/experiencia_vs_salario.png)

![Matriz de confusão](figures/matriz_confusao.png)

---

## 10. Discussão Técnica

O projeto apresenta uma metodologia coerente para um trabalho acadêmico de classificação. A principal decisão metodológica positiva é encapsular o pré-processamento em `Pipeline`, reduzindo risco de *data leakage*. Também é adequada a escolha de validação cruzada estratificada, pois a variável-alvo possui distribuição não uniforme.

A Regressão Logística funciona bem como modelo inicial por oferecer simplicidade, reprodutibilidade e boa interpretabilidade. No entanto, por ser um modelo linear, pode ter dificuldade para capturar interações relevantes, como a combinação entre senioridade, região, cargo e porte da empresa. Isso ajuda a explicar o desempenho mais baixo na classe média, onde os perfis profissionais são naturalmente mais sobrepostos.

---

## 11. Revisão do Projeto

### Pontos fortes

- README bem estruturado, com objetivo, metodologia, métricas e limitações.
- Notebook completo, organizado em etapas narrativas e técnicas.
- Uso correto de `Pipeline` e `ColumnTransformer`.
- Separação treino/teste estratificada.
- Discussão explícita sobre prevenção de *data leakage*.
- Figuras salvas em `reports/figures/`, facilitando auditoria e composição de relatório.
- Dados processados versionados em `data/processed/`, permitindo inspeção sem depender do dataset bruto.

### Ajustes aplicados nesta revisão

- Adicionado carregamento local de CSV em `data/raw/` antes da tentativa via KaggleHub. Isso melhora a execução em ambientes sem internet ou sem cache do Kaggle.
- Adicionada exportação explícita dos datasets processados no notebook.
- Convertidos `train_processed.csv` e `test_processed.csv` para UTF-8, permitindo leitura direta com `pandas.read_csv()` sem parâmetro extra de encoding.

### Pontos de atenção

- O projeto ainda não salva o modelo treinado em arquivo serializado (`.joblib` ou `.pkl`), então a inferência depende da reexecução do notebook.
- As dependências em `requirements.txt` não estão versionadas, o que pode gerar pequenas diferenças de execução em ambientes futuros.
- O notebook possui saídas salvas, mas o ambiente local atual não tinha Python/Scikit-learn disponível no PATH; a execução completa deve ser validada no Colab ou em ambiente com as dependências instaladas.
- O indicador `Perfil_Tecnico_Qtd` trata todas as tecnologias com o mesmo peso, embora algumas ferramentas possam ter impacto salarial maior que outras.

---

## 12. Limitações

O estudo utiliza dados declarados voluntariamente pelos respondentes, sujeitos a vieses de amostragem, autopreenchimento e representatividade. Além disso, a simplificação da variável-alvo em três classes reduz ruído, mas também perde granularidade salarial.

Outra limitação está na escolha do modelo. A Regressão Logística é adequada como baseline interpretável, mas modelos baseados em árvores, como Random Forest, Gradient Boosting, LightGBM ou XGBoost, podem capturar relações não lineares e interações complexas com maior naturalidade.

---

## 13. Melhorias Futuras

1. Testar modelos baseados em árvores e comparar métricas por classe.
2. Salvar o pipeline treinado com `joblib` para permitir inferência posterior.
3. Versionar dependências principais no `requirements.txt`.
4. Calcular importância de variáveis ou coeficientes do modelo para melhorar a explicabilidade.
5. Criar pesos ou grupos para tecnologias em vez de usar apenas a contagem simples.
6. Adicionar uma etapa opcional de avaliação com `balanced_accuracy` e F1 macro.
7. Transformar o notebook em script modular para facilitar reexecução automatizada.

---

## 14. Conclusão

O projeto está tecnicamente bem encaminhado e atende ao objetivo acadêmico de construir um pipeline supervisionado para prever faixa salarial de profissionais de dados no Brasil. O resultado de **72,46% de acurácia em teste** é consistente para um problema tabular com variáveis majoritariamente categóricas e com forte sobreposição entre perfis profissionais.

Como trabalho acadêmico, o projeto demonstra domínio das principais etapas de Ciência de Dados: entendimento do problema, seleção de variáveis, engenharia de atributos, prevenção de vazamento, validação cruzada, avaliação por métricas e análise crítica das limitações. As melhorias recomendadas são evoluções naturais para transformar o estudo em uma solução mais robusta, reprodutível e interpretável.
