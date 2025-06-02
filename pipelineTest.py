import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import random
from typing import List, Dict, Tuple
import torch
from tqdm import tqdm
import os

from transformers import BitsAndBytesConfig
from ragTest import MedicalSQLRetriever


from unsloth import FastLanguageModel
from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer, pipeline


DATASET_PATH = Path("text2sql_epi_dataset_omop.xlsx")
TEST_RATIO = 0.2
SCHEMA_PATH = "omop_schema_stub.txt"
TOP_K_SIMILAR = 2
RESULTS_TXT_PATH = "llm_sql_evaluation_results3.txt"


with open(SCHEMA_PATH, "r") as f:
    db_schema = f.read()


def split_dataset(dataset_path: Path, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df = pd.read_excel(dataset_path)
    indices = list(range(len(df)))
    random.seed(42) 
    random.shuffle(indices)
    
    split_idx = int(len(indices) * (1 - test_ratio))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[test_indices].reset_index(drop=True)


def build_rag_from_train(train_df: pd.DataFrame) -> MedicalSQLRetriever:
    temp_path = Path("temp_train_data.xlsx")
    train_df.to_excel(temp_path, index=False)
    
    retriever = MedicalSQLRetriever()
    retriever.build(temp_path)
    
    temp_path.unlink()
    
    return retriever

def generate_sql_with_model(model, tokenizer, prompt: str, max_tokens: int = 500):
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.95,
        )
    
    return tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)



def evaluate_models(test_df: pd.DataFrame, retriever: MedicalSQLRetriever, 
                   models_config: List[Dict], db_schema: str):
    results = []
    
    schema_truncated = "\n".join(db_schema.split("\n")[:50]) + "\n..."
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = row[test_df.columns[test_df.columns.str.upper().str.startswith('QUESTION')][0]]
        ground_truth_sql = row[test_df.columns[test_df.columns.str.upper().str.contains('RUNNABLE')][0]]
        
        similar_queries = retriever.query(question, k=TOP_K_SIMILAR)
        

        similar_examples = ""
        for i, (score, meta) in enumerate(similar_queries[:1], 1):
            similar_examples += f"Example {i}:\nQuestion: {meta['canonical_question']}\nSQL: {meta['sql']}\n\n"
        

        for model_config in models_config:
            model_name = model_config['name']
            model = model_config['model']
            tokenizer = model_config['tokenizer']
            
            prompt = f"""You are an expert epidemiology data analyst. Translate the question into SQL for OMOP CDM v5.3.

            ### Similar example
            {similar_examples}

            ### Question
            {question}

            ### SQL
            """
            
            try:
                generated_sql = generate_sql_with_model(model, tokenizer, prompt)
                
                results.append({
                    'question_id': idx,
                    'question': question,
                    'model': model_name,
                    'generated_sql': generated_sql,
                    'reference_question': question,
                    'reference_sql': ground_truth_sql,
                })
            except Exception as e:
                print(f"Error con el modelo {model_name} para la pregunta {idx}: {e}")
                results.append({
                    'question_id': idx,
                    'question': question,
                    'model': model_name,
                    'generated_sql': f"ERROR: {str(e)}",
                    'reference_question': question,
                    'reference_sql': ground_truth_sql,
                })
                
    return pd.DataFrame(results)


def save_results_to_txt(results_df: pd.DataFrame, output_path: str):

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUACIÓN DE MODELOS LLM PARA GENERACIÓN SQL\n")
        f.write("=" * 80 + "\n\n")
        

        for question_id in results_df['question_id'].unique():
            question_results = results_df[results_df['question_id'] == question_id]
            question = question_results['question'].iloc[0]
            reference_sql = question_results['reference_sql'].iloc[0]
            pregunta = question_results['reference_question'].iloc[0]
            
            f.write(f"PREGUNTA #{question_id}: {question}\n")
            f.write("-" * 80 + "\n")
            f.write("Pregunta DE REFERENCIA:\n")
            f.write(f"{question}\n\n")
            f.write("SQL DE REFERENCIA:\n")
            f.write(f"{reference_sql}\n\n")
            

            for _, row in question_results.iterrows():
                model_name = row['model']
                generated_sql = row['generated_sql']
                
                f.write(f"MODELO: {model_name}\n")
                f.write(f"SQL GENERADO:\n")
                f.write(f"{generated_sql}\n\n")
            
            f.write("=" * 80 + "\n\n")
        
        f.write("\nFIN DE LA EVALUACIÓN\n")

def main():
    print("Dividiendo el dataset en train/test...")
    train_df, test_df = split_dataset(DATASET_PATH, TEST_RATIO)
    print(f"Train: {len(train_df)} ejemplos, Test: {len(test_df)} ejemplos")
    
    print("Construyendo el índice RAG con el conjunto de entrenamiento...")
    retriever = build_rag_from_train(train_df)
    
    models_config = []
    
    # Modelo 1: sqlCoder-Qwen2.5-8bit
    print("Cargando el modelo sqlCoder-Qwen2.5-8bit...")
    model_name = "imsanjoykb/sqlCoder-Qwen2.5-8bit"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    models_config.append({
        'name': 'sqlCoder-Qwen2.5-8bit',
        'model': model,
        'tokenizer': tokenizer
    })
    
    # # Modelo 2: sqlcoder-7b-2
    # print("Cargando el modelo sqlcoder-7b-2...")
    # model_name_2 = "defog/sqlcoder-7b-2"
    # tokenizer_2 = AutoTokenizer.from_pretrained(model_name_2)
    # model_2 = AutoModelForCausalLM.from_pretrained(
    #     model_name_2,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     # Eliminar esta línea: load_in_4bit=True,
    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         llm_int8_enable_fp32_cpu_offload=True
    #     )
    # )
    # models_config.append({
    #     'name': 'sqlcoder-7b-2',
    #     'model': model_2,
    #     'tokenizer': tokenizer_2
    # })
    
    # # Modelo 3: hrida-t2sql-128k
    # print("Cargando el modelo hrida-t2sql-128k...")
    # model_name_3 = "Hrida/hrida-t2sql-128k"
    # tokenizer_3 = AutoTokenizer.from_pretrained(model_name_3)
    # model_3 = AutoModelForCausalLM.from_pretrained(
    #     model_name_3,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",

    #     quantization_config=BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         llm_int8_enable_fp32_cpu_offload=True
    #     )
    # )
    # models_config.append({
    #     'name': 'hrida-t2sql-128k',
    #     'model': model_3,
    #     'tokenizer': tokenizer_3
    # })
    

    print("Evaluando modelos...")
    results_df = evaluate_models(test_df, retriever, models_config, db_schema)
    

    results_df.to_csv("llm_sql_evaluation_results.csv", index=False)
    print("Resultados guardados en CSV: 'llm_sql_evaluation_results.csv'")
    

    save_results_to_txt(results_df, RESULTS_TXT_PATH)
    print(f"Resultados guardados en formato de texto: '{RESULTS_TXT_PATH}'")
    
    return results_df


if __name__ == "__main__":
    results = main()
    

    print("\nEjemplos de resultados:")
    print(results.head())