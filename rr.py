import os
from pdfminer.high_level import extract_text
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import spacy
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import streamlit as st
import pandas as pd
import re
import yake
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy import displacy

# Load NER models
ner_model = {
    "bert_portuguese": pipeline(
        "ner",
        model=AutoModelForTokenClassification.from_pretrained("lfcc/bert-portuguese-ner"),
        tokenizer=AutoTokenizer.from_pretrained("lfcc/bert-portuguese-ner"),
        aggregation_strategy="simple"
    ),
    "spacy": spacy.load("pt_core_news_lg")
}

# Label mapping
label_map = {
    "bert_portuguese": ["Organizacao", "Local", "Pessoa", "Profissao", "Data"],
    "spacy": ["ORG", "LOC", "PER", "MISC"]
}

# Functions for text extraction
extract_methods = {
    "pdfminer": lambda file_path: extract_text(file_path),
    "pydf": lambda file_path: "".join(page.extract_text() for page in PdfReader(file_path).pages),
    "pdfplumber": lambda file_path: "".join(page.extract_text() for page in pdfplumber.open(file_path).pages),
    "pytesseract": lambda file_path: "".join(pytesseract.image_to_string(page, lang='por') for page in convert_from_path(file_path))
}

def extract_text_sequentially(file_path):
    """Attempt to extract text from a PDF using different methods sequentially."""
    for method_name, method in extract_methods.items():
        try:
            print(f"Trying method: {method_name} for {file_path}")
            text = method(file_path)
            if text.strip():
                print(f"Text extracted successfully using {method_name}")
                return text, method_name
        except Exception as e:
            print(f"Method {method_name} failed: {e}")
    return None, None

def clean_text(text):
    """Perform basic text cleaning before applying NER."""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r'[^A-Za-z0-9á-úÁ-Ú\s]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    return text

def perform_keyword_extraction(text, n=1, num_keywords=10):
    """Perform keyword extraction on the text using YAKE and normalize scores using Min-Max normalization."""
    kw_extractor = yake.KeywordExtractor(lan="pt", n=n, top=num_keywords)
    keywords = kw_extractor.extract_keywords(text)
    
    scores = [score for _, score in keywords]
    
    min_score = min(scores)
    max_score = max(scores)
    
    if min_score == max_score:
        normalized_scores = [0.5 for _ in scores]  # Assign a neutral value if all scores are the same
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    
    return [(keyword, normalized_score) for (keyword, _), normalized_score in zip(keywords, normalized_scores)]

def make_word_cloud(text, max_words=50, n=1):
    """Create a word cloud from the text."""
    keyword2WordCloud = dict(perform_keyword_extraction(text, n=n, num_keywords=max_words))

    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color="white",  # Background color of the word cloud
        max_words=max_words,       # Maximum number of words to display
        colormap="viridis",        # Color scheme for the words
        stopwords=None,            # Optional: Add custom stopwords if needed
    ).generate_from_frequencies(keyword2WordCloud)

    plt.figure(figsize=(10, 5))  # Set the size of the figure
    plt.imshow(wordcloud, interpolation="bilinear")  # Display the word cloud
    plt.axis("off")  # Hide the axis
    st.pyplot(plt)  # Display the plot in Streamlit

def perform_ner_on_txt(text, model_name):
    """Perform Named Entity Recognition (NER) on the text using the specified model."""
    if not text.strip():
        return []

    if model_name == "bert_portuguese":
        ner_results = []
        chunk_size = 512  # BERT token limit
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            entities = ner_model[model_name](chunk)
            
            # Reconstruct words and assign the most common label
            current_word = ""
            current_labels = []
            for entity in entities:
                token = entity['word']
                label = entity['entity_group']
                
                if token.startswith("##"):
                    # Subword token: append to the current word
                    current_word += token[2:]
                    current_labels.append(label)
                else:
                    # New word: save the previous word and its label
                    if current_word:
                        # Assign the most common label to the word
                        most_common_label = max(set(current_labels), key=current_labels.count)
                        ner_results.append((current_word, most_common_label))
                    
                    # Start a new word
                    current_word = token
                    current_labels = [label]
            
            # Add the last word
            if current_word:
                most_common_label = max(set(current_labels), key=current_labels.count)
                ner_results.append((current_word, most_common_label))
        
        # Iterative matching of NER results with the text
        ents = []
        text_index = 0  # Tracks the current position in the text
        for word, label in ner_results:
            # Search for the word in the text starting from the current index
            word_start = text.find(word, text_index)
            if word_start == -1:
                print(f"Word '{word}' not found in text after index {text_index}")
                continue  # Skip this word if it's not found
            
            word_end = word_start + len(word)
            ents.append({"start": word_start, "end": word_end, "label": label})
            
            # Move the text index forward to avoid overlapping matches
            text_index = word_end
        
        return (ner_results, {"text": text, "ents": ents, "title": None})

    if model_name == "spacy":
        doc = ner_model[model_name](text)
        return ([(ent.text, ent.label_) for ent in doc.ents], doc)

    return []

def filter_entities(entities, model_name, selected_labels):
    """Filter entities based on user-selected labels."""
    valid_labels = label_map[model_name]
    return [(word, label) for word, label in entities if label in valid_labels and label in selected_labels]

def display_tokens_and_entities(text, model_name, ner_results, doc):
    """Display tokens and highlight entities using the selected model."""
    if model_name == "spacy":
        html = displacy.render(doc, style="ent", page=True)
    elif model_name == "bert_portuguese":
        # Convert BERT results to spaCy-like format for visualization
        html = displacy.render(doc, style="ent", manual=True, page=True)
    else:
        st.warning("Modelo não suportado para visualização.")
        return

    st.components.v1.html(html, height=1000, scrolling=True)

def process_text(text):
    """Process the text and display the results."""
    if text:
        # Display the original text
        st.subheader("Texto Original")
        st.text_area("Texto Original", text, height=300)

        # Clean the text
        cleaned_text = clean_text(text)

        # Display the cleaned text
        st.subheader("Texto Limpo")
        st.text_area("Texto Limpo", cleaned_text, height=300)

        # Add a slider to adjust the number of words in the word cloud
        max_words = st.slider(
            "Número máximo de palavras no Word Cloud:",
            min_value=10,
            max_value=200,
            value=50,  # Default value
            step=10,
            key="max_words"
        )

        n_value = st.slider(
            "Tamanho do n-grama (n):",
            min_value=1,
            max_value=15,
            value=1, 
            step=1,
            key="n_value"
        )

        if "prev_max_words" not in st.session_state:
            st.session_state.prev_max_words = max_words
        if "prev_n_value" not in st.session_state:
            st.session_state.prev_n_value = n_value

        if (st.session_state.prev_max_words != max_words) or (st.session_state.prev_n_value != n_value):
            st.session_state.prev_max_words = max_words
            st.session_state.prev_n_value = n_value
            with st.spinner("Gerando Word Cloud..."):
                make_word_cloud(cleaned_text, max_words=max_words, n=n_value)

        if st.button("Gerar Word Cloud"):
            with st.spinner("Gerando Word Cloud..."):
                make_word_cloud(cleaned_text, max_words=max_words, n=n_value)

        # Seleção do modelo NER
        model_name = st.selectbox("Selecione o modelo para NER:", ["bert_portuguese", "spacy"])

        # Seleção de labels
        selected_labels = st.multiselect(
            "Selecione os rótulos que deseja filtrar:",
            options=label_map[model_name],
            default=label_map[model_name]
        )

        if st.button("Executar NER"):
            with st.spinner("Executando NER..."):
                ner_results, doc = perform_ner_on_txt(cleaned_text, model_name)

            if ner_results:
                st.subheader("Resultados de NER")
                ner_table = [{"Palavra": word, "Entidade": entity} for word, entity in ner_results]
                st.table(ner_table)

                # Display tokens and highlighted entities
                st.subheader("Tokens e Entidades Destacadas")
                display_tokens_and_entities(cleaned_text, model_name, ner_results, doc)
            else:
                st.warning("Nenhuma entidade foi encontrada no texto.")
    else:
        st.warning("Falha ao extrair texto do documento.")

def carregar_textos(diretorio_txts):
    """Carrega todos os textos dos arquivos .txt dentro das pastas de anos."""
    textos = {}
    for ano in range(2021, 2026):
        pasta_ano = os.path.join(diretorio_txts, str(ano))
        if os.path.isdir(pasta_ano):
            for arquivo in os.listdir(pasta_ano):
                if arquivo.endswith(".txt"):
                    caminho_arquivo = os.path.join(pasta_ano, arquivo)
                    with open(caminho_arquivo, "r", encoding="utf-8") as f:
                        textos[f"{ano}: {arquivo}"] = f.read()
    return textos

def main():
    st.set_page_config(page_title="Citylink - Atas de Reuniões Camarárias", layout="wide")
    
    # Diretório base dos textos
    diretorio_txts = "diretorio_txts"
    
    # Carregar textos automaticamente
    textos_carregados = carregar_textos(diretorio_txts)
    st.session_state.example_results = {key: {"text": value, "cleaned_text": value} for key, value in textos_carregados.items()}
    st.session_state.setdefault("prev_file_name", None)
    st.session_state.setdefault("prev_text", None)

    # Sidebar para navegação
    st.sidebar.title("Navegação")
    page = st.sidebar.radio("Ir para:", ("Página Inicial", "Exemplos de Funcionamento"))

    if page == "Página Inicial":
        st.title("Citylink - Atas de Reuniões Camarárias")
        st.header("📊 Análise de documentos")
        st.write("Explore os pontos chave retirados do documento!")
        
        uploaded_file = st.file_uploader("Carregar Documento de Texto ou PDF", type=["txt", "pdf"])
        if uploaded_file:

            
            file_name = uploaded_file.name
            print("------------------------", st.session_state.prev_file_name)

            if ( st.session_state.prev_file_name is not None) and st.session_state.prev_file_name == file_name:
                text = st.session_state.prev_text
            else:
            
                # Processa texto ou PDF
                with st.spinner("Processando o documento..."):
                    if uploaded_file.name.endswith(".txt"):
                        text = uploaded_file.read().decode("utf-8")
                    else:
                        # Salva o arquivo temporariamente
                        file_path = f"uploaded_{uploaded_file.name}"
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.read())

                        # Extrai o texto utilizando o método apropriado
                        text, method_name = extract_text_sequentially(file_path)
                        os.remove(file_path)
                        
                st.session_state.prev_file_name = file_name
                st.session_state.prev_text = text
            
            if text:
                process_text(text)

            
        else:
            st.info("Faça upload de um ficheiro para realizar a análise.")
    
    elif page == "Exemplos de Funcionamento":
        st.title("Exemplos de Funcionamento")
        selected_example = st.sidebar.selectbox("Selecione um exemplo:", list(st.session_state.example_results.keys()))
        
        if selected_example:
            results = st.session_state.example_results[selected_example]
            st.subheader(f"📄 {selected_example}")
            process_text(results["text"])

if __name__ == "__main__":
    main()