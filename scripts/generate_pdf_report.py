from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
OUTPUT_PDF = REPORTS_DIR / "RELATORIO_TECNICO_STATE_OF_DATA.pdf"

TITLE = "PREDIÇÃO DE FAIXA SALARIAL DE PROFISSIONAIS DE DADOS NO BRASIL COM MACHINE LEARNING"
AUTHORS = "RODOLFO VINICIUS CIMA TAKEMOTO<br/>TIAGO GALHARDO AVELAR"
INSTITUTION = "FACULDADE DE TECNOLOGIA DE JUNDIAÍ"
COURSE = "INTELIGÊNCIA COMPUTACIONAL"
PROFESSOR = "Me. Mateus Guilherme Fuini"
CITY = "Jundiaí"
YEAR = "2026"


class ABNTDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
            showBoundary=0,
        )
        self.addPageTemplates([PageTemplate(id="abnt", frames=[frame], onPage=draw_page_number)])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable, "_toc_level"):
            key = getattr(flowable, "_bookmark_name", None)
            text = flowable.getPlainText()
            if key:
                self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (flowable._toc_level, text, self.page, key))


def draw_page_number(canvas, doc):
    # ABNT conta os elementos pre-textuais, mas exibe a numeracao apenas apos eles.
    if doc.page <= 3:
        return

    canvas.saveState()
    canvas.setFont("Times-Roman", 10)
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 2 * cm + 0.2 * cm, str(doc.page))
    canvas.restoreState()


def make_styles():
    return {
        "cover_top": ParagraphStyle(
            "cover_top",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "cover_author": ParagraphStyle(
            "cover_author",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            firstLineIndent=1.25 * cm,
            spaceAfter=0,
        ),
        "body_no_indent": ParagraphStyle(
            "body_no_indent",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=0,
        ),
        "small": ParagraphStyle(
            "small",
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            alignment=TA_JUSTIFY,
            firstLineIndent=0,
            spaceAfter=0,
        ),
        "center": ParagraphStyle(
            "center",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
        "right_block": ParagraphStyle(
            "right_block",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_JUSTIFY,
            leftIndent=7.0 * cm,
            firstLineIndent=0,
            spaceAfter=0,
        ),
        "heading1": ParagraphStyle(
            "heading1",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_LEFT,
            spaceBefore=12,
            spaceAfter=12,
            firstLineIndent=0,
        ),
        "heading2": ParagraphStyle(
            "heading2",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_LEFT,
            spaceBefore=6,
            spaceAfter=6,
            firstLineIndent=0,
        ),
        "caption_title": ParagraphStyle(
            "caption_title",
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "caption_source": ParagraphStyle(
            "caption_source",
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            alignment=TA_LEFT,
            spaceBefore=3,
            spaceAfter=12,
        ),
        "reference": ParagraphStyle(
            "reference",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_LEFT,
            firstLineIndent=0,
            spaceAfter=6,
        ),
        "toc_title": ParagraphStyle(
            "toc_title",
            fontName="Times-Bold",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "toc_entry": ParagraphStyle(
            "toc_entry",
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            leftIndent=0,
            firstLineIndent=0,
        ),
    }


def paragraph(text, style):
    return Paragraph(text, style)


def heading(text, style, level, key):
    flowable = Paragraph(text, style)
    flowable._toc_level = level
    flowable._bookmark_name = key
    return flowable


def bullets(items, styles):
    return ListFlowable(
        [ListItem(paragraph(item, styles["body_no_indent"]), leftIndent=0) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=1.25 * cm,
        bulletFontName="Times-Roman",
        bulletFontSize=8,
    )


def table_title(text, styles):
    return paragraph(text, styles["caption_title"])


def source(text, styles):
    return paragraph(f"Fonte: {text}", styles["caption_source"])


def abnt_table(data, widths_cm, styles, title, source_text="Elaborado pelos autores (2026)."):
    converted = []
    for row in data:
        converted.append([Paragraph(str(cell), styles["small"]) for cell in row])

    tbl = Table(converted, colWidths=[width * cm for width in widths_cm], hAlign="CENTER", repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Times-Roman"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("LEADING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EFEFEF")),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([table_title(title, styles), tbl, source(source_text, styles)])


def scaled_image(path, max_width, max_height):
    with PILImage.open(path) as img:
        width_px, height_px = img.size

    ratio = min(max_width / width_px, max_height / height_px)
    return Image(str(path), width=width_px * ratio, height=height_px * ratio, hAlign="CENTER")


def figure(filename, title, styles):
    image_path = FIGURES_DIR / filename
    return KeepTogether(
        [
            paragraph(title, styles["caption_title"]),
            scaled_image(image_path, max_width=15.5 * cm, max_height=12.0 * cm),
            source("Elaborado pelos autores com base no State of Data Brazil 2024-2025.", styles),
        ]
    )


def cover(styles):
    return [
        paragraph(INSTITUTION, styles["cover_top"]),
        paragraph("CURSO DE TECNOLOGIA", styles["cover_top"]),
        Spacer(1, 4.0 * cm),
        paragraph(AUTHORS, styles["cover_author"]),
        Spacer(1, 5.0 * cm),
        paragraph(TITLE, styles["cover_title"]),
        Spacer(1, 6.6 * cm),
        paragraph(CITY, styles["cover_top"]),
        paragraph(YEAR, styles["cover_top"]),
        PageBreak(),
    ]


def title_page(styles):
    return [
        paragraph(AUTHORS, styles["cover_author"]),
        Spacer(1, 4.7 * cm),
        paragraph(TITLE, styles["cover_title"]),
        Spacer(1, 3.5 * cm),
        paragraph(
            "Relatório técnico apresentado à disciplina de Inteligência Computacional, da Faculdade "
            "de Tecnologia de Jundiaí, como parte das atividades acadêmicas relacionadas à aplicação "
            "de técnicas de Aprendizado de Máquina supervisionado.",
            styles["right_block"],
        ),
        Spacer(1, 0.7 * cm),
        paragraph(f"Professor: {PROFESSOR}", styles["right_block"]),
        Spacer(1, 5.5 * cm),
        paragraph(CITY, styles["cover_top"]),
        paragraph(YEAR, styles["cover_top"]),
        PageBreak(),
    ]


def abstract_page(styles):
    return [
        paragraph("<b>RESUMO</b>", styles["center"]),
        Spacer(1, 0.8 * cm),
        paragraph(
            "Este trabalho apresenta o desenvolvimento de um pipeline de Aprendizado de Máquina "
            "supervisionado para classificar a faixa salarial de profissionais brasileiros da área "
            "de dados. A base utilizada foi a pesquisa State of Data Brazil 2024-2025, composta "
            "originalmente por 5.217 respondentes e mais de 400 variáveis. O estudo selecionou "
            "atributos demográficos, acadêmicos, profissionais e tecnológicos, além de variáveis "
            "derivadas relacionadas à experiência, ao trabalho remoto e à quantidade de tecnologias "
            "utilizadas. A variável-alvo foi simplificada em três classes: baixa, média e alta faixa "
            "salarial. O modelo final adotado foi uma Regressão Logística otimizada por validação "
            "cruzada estratificada, alcançando 73,47% de acurácia média na validação cruzada e 72,46% "
            "de acurácia no conjunto de teste. Os resultados indicam melhor desempenho nas classes "
            "salariais extremas e maior dificuldade na classificação da faixa média, o que é coerente "
            "com a sobreposição de perfis profissionais no mercado brasileiro de dados.",
            styles["body"],
        ),
        Spacer(1, 0.5 * cm),
        paragraph(
            "<b>Palavras-chave:</b> Aprendizado de Máquina; Ciência de Dados; Classificação; "
            "Faixa Salarial; State of Data.",
            styles["body_no_indent"],
        ),
        PageBreak(),
    ]


def toc_page(styles):
    toc = TableOfContents()
    toc.levelStyles = [styles["toc_entry"], styles["toc_entry"]]
    return [paragraph("<b>SUMÁRIO</b>", styles["toc_title"]), toc, PageBreak()]


def build_story(styles):
    story = []
    story.extend(cover(styles))
    story.extend(title_page(styles))
    story.extend(abstract_page(styles))
    story.extend(toc_page(styles))

    story.append(heading("1 INTRODUÇÃO", styles["heading1"], 0, "sec-introducao"))
    story.append(
        paragraph(
            "A área de dados no Brasil reúne profissionais com diferentes trajetórias, formações, "
            "cargos, níveis de senioridade e contextos de contratação. Essa diversidade torna a "
            "análise salarial um problema relevante tanto para profissionais que buscam orientar suas "
            "carreiras quanto para empresas que desejam estruturar políticas de remuneração mais "
            "compatíveis com o mercado.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "O presente trabalho tem como objetivo construir, otimizar e avaliar um modelo de "
            "Aprendizado de Máquina supervisionado capaz de classificar profissionais de dados em "
            "três faixas salariais: baixa, média e alta. Para isso, utiliza-se a pesquisa State of "
            "Data Brazil 2024-2025, uma das principais fontes públicas sobre o mercado brasileiro "
            "de dados.",
            styles["body"],
        )
    )

    story.append(heading("2 FUNDAMENTAÇÃO E CARACTERIZAÇÃO DO PROBLEMA", styles["heading1"], 0, "sec-fundamentacao"))
    story.append(
        paragraph(
            "O problema tratado é caracterizado como uma tarefa de classificação multiclasse. A "
            "variável de interesse corresponde à faixa salarial declarada pelos respondentes, enquanto "
            "as variáveis preditoras representam aspectos demográficos, acadêmicos, profissionais e "
            "tecnológicos. Entre os fatores considerados estão senioridade, experiência, região, "
            "formação, setor econômico, tamanho da empresa, modelo de trabalho e cargo exercido.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "A opção por classificar faixas salariais, em vez de prever valores monetários exatos, "
            "é adequada ao formato do questionário original e reduz a sensibilidade do modelo a "
            "variações individuais extremas. Além disso, o agrupamento da variável-alvo em três classes "
            "diminui a esparsidade e favorece uma análise mais estável do desempenho preditivo.",
            styles["body"],
        )
    )

    story.append(heading("3 MATERIAIS E MÉTODOS", styles["heading1"], 0, "sec-metodos"))
    story.append(heading("3.1 Base de dados", styles["heading2"], 1, "sec-base"))
    story.append(
        paragraph(
            "A base utilizada foi o State of Data Brazil 2024-2025, promovido pela comunidade Data "
            "Hackers em parceria com a Bain & Company. O arquivo bruto possui originalmente 5.217 "
            "respondentes e mais de 400 variáveis, incluindo dados demográficos, formação acadêmica, "
            "situação profissional, cargo, senioridade, remuneração e uso cotidiano de tecnologias.",
            styles["body"],
        )
    )
    story.append(
        abnt_table(
            [["Partição", "Instâncias"], ["Treino", "3.890"], ["Teste", "973"], ["Total modelado", "4.863"]],
            [8.0, 4.0],
            styles,
            "Tabela 1 - Distribuição das instâncias após tratamento",
            "Elaborado pelos autores (2026).",
        )
    )
    story.append(
        abnt_table(
            [
                ["Classe no conjunto de teste", "Instâncias"],
                ["Alta faixa salarial", "368"],
                ["Média faixa salarial", "347"],
                ["Baixa faixa salarial", "258"],
            ],
            [8.0, 4.0],
            styles,
            "Tabela 2 - Distribuição das classes no conjunto de teste",
            "Elaborado pelos autores (2026).",
        )
    )

    story.append(heading("3.2 Variável-alvo", styles["heading2"], 1, "sec-alvo"))
    story.append(
        paragraph(
            "A variável original de salário contém 13 intervalos de renda mensal. Para viabilizar "
            "uma classificação mais robusta, esses intervalos foram agrupados em três classes: baixa "
            "faixa salarial, média faixa salarial e alta faixa salarial.",
            styles["body"],
        )
    )
    story.append(
        abnt_table(
            [
                ["Intervalo original", "Classe simplificada"],
                ["Até R$ 6.000/mês", "Baixa faixa salarial"],
                ["De R$ 6.001/mês a R$ 12.000/mês", "Média faixa salarial"],
                ["Acima de R$ 12.000/mês", "Alta faixa salarial"],
            ],
            [8.0, 6.0],
            styles,
            "Tabela 3 - Agrupamento da variável-alvo",
            "Elaborado pelos autores (2026).",
        )
    )

    story.append(heading("3.3 Seleção e engenharia de atributos", styles["heading2"], 1, "sec-atributos"))
    story.append(
        paragraph(
            "O conjunto final de preditores foi composto por variáveis relacionadas ao perfil do "
            "respondente e ao contexto de atuação profissional. Foram utilizadas variáveis como gênero, "
            "escolaridade, região, situação de trabalho, setor, tamanho da empresa, cargo, nível de "
            "senioridade, experiência categorizada, trabalho remoto e quantidade de tecnologias.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "Foram criadas três variáveis derivadas: Experiencia_Categoria, que agrupa o tempo de "
            "atuação em dados; Trabalha_Remoto, que identifica atuação 100% remota; e Perfil_Tecnico_Qtd, "
            "que representa a soma das tecnologias utilizadas no cotidiano.",
            styles["body"],
        )
    )

    story.append(heading("3.4 Pipeline de modelagem", styles["heading2"], 1, "sec-pipeline"))
    story.append(
        paragraph(
            "O pré-processamento foi encapsulado por meio de ColumnTransformer e Pipeline, evitando "
            "vazamento de dados entre treino e teste. A variável numérica Perfil_Tecnico_Qtd recebeu "
            "imputação por mediana, tratamento de outliers por capping baseado em IQR e padronização "
            "com StandardScaler. As variáveis categóricas receberam imputação por moda e codificação "
            "OneHotEncoder com tratamento de categorias desconhecidas.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "O modelo final foi uma Regressão Logística. Os hiperparâmetros foram ajustados com "
            "GridSearchCV, utilizando validação cruzada estratificada com cinco partições. A métrica "
            "de otimização foi a acurácia.",
            styles["body"],
        )
    )

    story.append(heading("4 RESULTADOS E DISCUSSÃO", styles["heading1"], 0, "sec-resultados"))
    story.append(
        abnt_table(
            [
                ["Métrica", "Resultado"],
                ["Acurácia média na validação cruzada", "73,47%"],
                ["Desvio padrão da validação cruzada", "1,69%"],
                ["Acurácia no conjunto de teste", "72,46%"],
            ],
            [9.0, 4.0],
            styles,
            "Tabela 4 - Métricas gerais do modelo",
            "Elaborado pelos autores (2026).",
        )
    )
    story.append(
        abnt_table(
            [
                ["Classe", "Precisão", "Recall", "F1-score", "Suporte"],
                ["Alta faixa salarial", "0,76", "0,82", "0,79", "368"],
                ["Baixa faixa salarial", "0,81", "0,73", "0,77", "258"],
                ["Média faixa salarial", "0,63", "0,62", "0,62", "347"],
            ],
            [5.0, 2.2, 2.0, 2.0, 2.0],
            styles,
            "Tabela 5 - Relatório de classificação no conjunto de teste",
            "Elaborado pelos autores (2026).",
        )
    )
    story.append(
        paragraph(
            "Os resultados indicam desempenho mais consistente nas classes extremas. A classe Alta "
            "faixa salarial apresentou F1-score de 0,79, enquanto a classe Baixa faixa salarial "
            "apresentou F1-score de 0,77. A classe Média faixa salarial obteve F1-score de 0,62, "
            "indicando maior dificuldade de separação entre os perfis.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "A maior ambiguidade na faixa média é compatível com o mercado brasileiro de dados, no "
            "qual profissionais plenos, seniores, especialistas e gestores podem ter perfis próximos, "
            "mas remunerações diferentes conforme região, porte da empresa, regime de contratação e "
            "cargo exercido.",
            styles["body"],
        )
    )

    figures = [
        ("distribuicao_alvo.png", "Figura 1 - Distribuição da variável-alvo simplificada"),
        ("senioridade_vs_salario.png", "Figura 2 - Faixa salarial por nível de senioridade"),
        ("experiencia_vs_salario.png", "Figura 3 - Distribuição salarial por categoria de experiência"),
        ("distribuicao_tecnologias.png", "Figura 4 - Distribuição da stack técnica por faixa salarial"),
        ("boxplot_tecnologias_por_faixa.png", "Figura 5 - Boxplot da quantidade de tecnologias por faixa salarial"),
        ("heatmap_correlacao.png", "Figura 6 - Heatmap exploratório de correlação"),
        ("matriz_confusao.png", "Figura 7 - Matriz de confusão do modelo preditivo"),
    ]
    for filename, title in figures:
        story.append(Spacer(1, 0.4 * cm))
        story.append(figure(filename, title, styles))

    story.append(heading("5 LIMITAÇÕES", styles["heading1"], 0, "sec-limitacoes"))
    story.append(
        paragraph(
            "O estudo utiliza respostas voluntárias declaradas pelos participantes, o que pode "
            "introduzir vieses de amostragem, autopreenchimento e representatividade. Além disso, "
            "a simplificação da variável-alvo em três classes reduz ruído, mas também elimina parte "
            "da granularidade salarial presente no questionário original.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "Outra limitação está na escolha do modelo. A Regressão Logística é adequada como modelo "
            "interpretável e baseline acadêmico, porém possui menor capacidade de capturar relações "
            "não lineares e interações complexas entre variáveis, quando comparada a modelos baseados "
            "em árvores.",
            styles["body"],
        )
    )

    story.append(heading("6 CONCLUSÃO", styles["heading1"], 0, "sec-conclusao"))
    story.append(
        paragraph(
            "O projeto atingiu o objetivo de construir e avaliar um pipeline supervisionado para "
            "classificar a faixa salarial de profissionais de dados no Brasil. A acurácia de 72,46% "
            "no conjunto de teste indica desempenho consistente para um problema tabular com variáveis "
            "majoritariamente categóricas e com forte sobreposição entre perfis profissionais.",
            styles["body"],
        )
    )
    story.append(
        paragraph(
            "Como trabalho acadêmico, o projeto demonstra domínio das etapas de Ciência de Dados, "
            "incluindo entendimento do problema, seleção de variáveis, engenharia de atributos, "
            "prevenção de vazamento de dados, validação cruzada, avaliação por métricas e análise "
            "crítica das limitações. Como continuidade, recomenda-se testar modelos baseados em árvores, "
            "salvar o pipeline treinado e ampliar a explicabilidade dos atributos utilizados.",
            styles["body"],
        )
    )

    story.append(heading("REFERÊNCIAS", styles["heading1"], 0, "sec-referencias"))
    references = [
        "DATA HACKERS; BAIN & COMPANY. <b>State of Data Brazil 2024-2025</b>. Kaggle, 2025. Disponível em: https://www.kaggle.com/datasets/datahackers/state-of-data-brazil-20242025. Acesso em: 27 maio 2026.",
        "HUNTER, John D. Matplotlib: A 2D graphics environment. <b>Computing in Science & Engineering</b>, v. 9, n. 3, p. 90-95, 2007.",
        "MCKINNEY, Wes. Data structures for statistical computing in Python. In: <b>Proceedings of the 9th Python in Science Conference</b>. Austin: SciPy, 2010. p. 56-61.",
        "PEDREGOSA, Fabian et al. Scikit-learn: Machine learning in Python. <b>Journal of Machine Learning Research</b>, v. 12, p. 2825-2830, 2011.",
    ]
    for item in references:
        story.append(paragraph(item, styles["reference"]))

    return story


def build_pdf():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    doc = ABNTDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=3 * cm,
        topMargin=3 * cm,
        bottomMargin=2 * cm,
        title=TITLE,
        author="Rodolfo Vinicius Cima Takemoto; Tiago Galhardo Avelar",
        subject="Relatório técnico em formato ABNT",
    )
    story = build_story(styles)
    doc.multiBuild(story)
    return OUTPUT_PDF


if __name__ == "__main__":
    output = build_pdf()
    print(output)
