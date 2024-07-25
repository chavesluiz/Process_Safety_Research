# Importação das bibliotecas

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Função para carregar os artigos contidos no arquivo .RIS obtido da base de Dados ScienceDirect and Scopus
def load_ris(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    records = []
    record = {}
    for line in lines:
        if line.startswith('TY'):
            if record:
                records.append(record)
            record = {}
        if line.strip() == '':
            continue
        key, value = line[:2], line[6:].strip()
        if key in record:
            record[key].append(value)
        else:
            record[key] = [value]
    if record:
        records.append(record)
    
    return records

# Função para extrair as informações desejadas dos registros do file .RIS
def extract_info(records):
    data = []
    for record in records:
        title = ' '.join(record.get('TI', ''))
        authors = ', '.join(record.get('AU', ''))
        year = ' '.join(record.get('PY', ''))
        source = ' '.join(record.get('T2', ''))
        abstract = ' '.join(record.get('AB', ''))
        data.append({'title': title, 'authors': authors, 'year': year, 'source': source, 'abstract': abstract})
    return pd.DataFrame(data)

# Função para calcular a similaridade semântica usando BERT

def calculate_similarity(texts, reference_texts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    def embed_text(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    reference_embeddings = np.vstack([embed_text(ref) for ref in reference_texts])
    text_embeddings = np.vstack([embed_text(text) for text in texts])

    similarities = cosine_similarity(text_embeddings, reference_embeddings)
    return similarities.max(axis=1)

# Função principal para carregar o file .RIS e salvar os artigod em planilha Excel para cada grupo
def main():
    # Carregar o arquivo .RIS
    file_path = 'D:/0000000HD_E/_________WORKS 2024/UERJ/_Ações/+_24_JULHO/Artigos_FINAL.ris'
    records = load_ris(file_path)

# Extrair os informações desejadas
    df = extract_info(records)
    
# Especificar cada grupo de informações desejadas conforme o contexto
# Textos de referência para comparação semântica

    # File 12 xlsx: reference_texts = ["teaching of occupational and health for accidents analysis"]
    # File 13 xlsx: reference_texts = ["teaching of process safety for accidents analysis in chemical industries"]
    # File 14 xlsx:reference_texts = ["process safety for chemical engineering and accident prevention in chemical industry"]
    # File 15 xlsx:reference_texts = ["curriculum of process safety and safetey education for chemical engineering"]
    # File 16 xlsx:reference_texts = ["occupational risks and occupation health for risk assessment and evaluetion safety"]

    # File 17 xlsx:reference_texts = ["occupational risks and occupational health for accidents risk management"]

    # File 18 xlsx:reference_texts = ["process safety with inherently safer design acident prevention in chemical industry"]
    # File 19 xlsx:reference_texts = ["environmental impact of industrial activities concern for sustainability for impact prevention and environmental protection"]
    # File 20 xlsx:reference_texts = ["loss prevention and accident prevention for safety engineering in process safety "]
    # File 21 xlsx:reference_texts = ["program of american institute of chemical (AIChe) for safety "]
    # File 22 reference_texts = ["safety education is a component of the curriculum in chemical engineering"]
    reference_texts = ["process dynamics and simulation are used for safety education"]
    
# Calcular similaridade
    texts = df['title'] + ' ' + df['abstract']
    similarities = calculate_similarity(texts, reference_texts)

# Adicionar coluna de similaridade ao DataFrame
    df['similarity'] = similarities

# Classificar os artigos como aderentes ou não aderentes com especificação de critério de 70% de similaridade
    df['adherent'] = similarities > 0.7

# Salvar os resultados em um arquivo Excel
    with pd.ExcelWriter('D:/0000000HD_E/_________WORKS 2024/UERJ/_Ações/+_24_JULHO/classified_articles_23.xlsx') as writer:
        df.to_excel(writer, sheet_name='All_Articles', index=False)

if __name__ == "__main__":
    main()
