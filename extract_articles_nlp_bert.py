# Import of libraries

import pandas as pd # For data manipulation
from transformers import BertTokenizer, BertModel # For use NLP processing with BERT
import torch # For computing with PyTorch
from sklearn.metrics.pairwise import cosine_similarity # Calculation of similarity with the cosine function
import numpy as np # Performing numerical operations

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

# Função para calcular a similaridade semântica usando arquitetura BERT

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

# Função principal para carregar o file .RIS e salvar os artigo em planilha Excel para cada grupo
def main():
    # Load file .RIS with articles - abstracts 
    file_path = 'D:/0000000HD_E/WORKS_2023/UERJ/Resultados/Artigos_FINAL.ris'
    records = load_ris(file_path)

# Extrair os informações desejadas
    df = extract_info(records)
    
# Specify each group of desired information according to the context
# Reference texts for semantic comparison
# Multiple inputs for execution. Just remove the beginning of each line to execute the command:
# Example: "# File i xlsx:" for multiple choice sentences

    # File 11 xlsx:reference_texts = ["Teaching of Process Safety in Chemical Engineering Course"]
    # File 11A xlsx:reference_texts = ["Safety education and process safety in Chemical Engineering Course"]
    # File 12 xlsx: reference_texts = ["teaching of occupational and health for accidents analysis"]
    # File 12_A xlsx:reference_texts = ["Teaching of occupational and health for process safety"]
    # File 13 xlsx: reference_texts = ["teaching of process safety for accidents analysis in chemical industries"]
    # File 14 xlsx:reference_texts = ["process safety for chemical engineering and accident prevention in chemical industry"]
    # File 15 xlsx:reference_texts = ["curriculum of process safety and safety education for chemical engineering"]
    # File 15A xlsx: reference_texts = ["Curriculum of process safety and safety education for chemical engineering"]
    # File 16 xlsx:reference_texts = ["occupational risks and occupation health for risk assessment and evaluetion safety"]

    # File 17 xlsx:reference_texts = ["occupational risks and occupational health for accidents risk management"]

    # File 18 xlsx:reference_texts = ["Process safety with inherently safer design accident prevention in chemical industry"]
    # File 18 C xlsx: reference_texts = ["Teaching inherently safer design for loss prevention."]
    # File 19 xlsx:reference_texts = ["Environmental impact of industrial activities concern for sustainability for impact prevention and environmental protection"]
    # File 20 xlsx:reference_texts = ["Loss prevention and accident prevention for safety engineering in process safety "]
    # File 21 xlsx:reference_texts = ["Program of american institute of chemical Engineering (AIChe) for safety "]
    # File 21A xlsx:reference_texts = ["Program of American institute of chemical Engineering (AIChE)"]
    # File 22 xlsx: reference_texts = ["Safety education is a component of the curriculum in chemical engineering"]
    # File 23 xlsx: reference_texts = ["Process dynamics and simulation are used for safety education"]
    # File 24 xlsx:reference_texts = ["Dynamic simulation and mathematical models are used for safety education"]
    # File 25 xlsx: reference_texts = ["Process dynamic and mathematical models for safety education"]
    # File 28 xlsx: reference_texts = ["Safety engineering for teaching of loss prevention."]
    
# Calculate cosine similarity for verify adhereny 
    texts = df['title'] + ' ' + df['abstract']
    similarities = calculate_similarity(texts, reference_texts)

# Add column and similarity in Excel file
    df['similarity'] = similarities

# Classify articles as compliant or non-compliant with a 70% similarity criterion specification
    df['adherent'] = similarities > 0.7

# Salvar os resultados no arquivo Excel na pasta de trabalho
# Save the results to the Excel file in the workbook
    with pd.ExcelWriter('D:/0000000HD_E/WORKS_2023/UERJ/Resultados/classified_articles_28.xlsx') as writer:
        df.to_excel(writer, sheet_name='All_Articles', index=False)

if __name__ == "__main__":
    main()

