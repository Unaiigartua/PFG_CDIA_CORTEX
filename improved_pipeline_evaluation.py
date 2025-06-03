import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm
import time

from ragTest import MedicalSQLRetriever
from omop_testing.test_sql_execution import SQLExecutabilityTester
from unsloth import FastLanguageModel

DATASET_PATH = Path("text2sql_epi_dataset_omop.xlsx")
OMOP_DB_PATH = Path("omop_testing/omop_test.db")
SCHEMA_PATH = "omop_schema_stub.txt"
TEST_RATIO = 0.2
TOP_K_SIMILAR = 1
MAX_ITERATIONS = 3
MAX_TOKENS = 500
RESULTS_DIR = Path("evaluation_results")

with open(SCHEMA_PATH, "r") as f:
    db_schema = f.read()

class ImprovedSQLEvaluator:
    
    def __init__(self):
        self.sql_tester = SQLExecutabilityTester(str(OMOP_DB_PATH))
        self.retriever = None
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
        
    def split_dataset(self, dataset_path: Path, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_excel(dataset_path)
        indices = list(range(len(df)))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(len(indices) * (1 - test_ratio))
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        return df.iloc[train_indices].reset_index(drop=True), df.iloc[test_indices].reset_index(drop=True)
    
    def build_rag_from_train(self, train_df: pd.DataFrame) -> MedicalSQLRetriever:
        temp_path = Path("temp_train_data.xlsx")
        train_df.to_excel(temp_path, index=False)
        
        retriever = MedicalSQLRetriever()
        retriever.build(temp_path)
        
        temp_path.unlink()
        return retriever
    
    def load_model(self, model_name: str) -> Tuple[any, any]:
        
        try:
            if "sqlCoder-Qwen2.5-8bit" in model_name:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name="imsanjoykb/sqlCoder-Qwen2.5-8bit",
                    max_seq_length=1024,
                    load_in_4bit=True,
                )
                FastLanguageModel.for_inference(model)
                
            elif "sqlcoder-7b-2" in model_name:
                from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                
                tokenizer = AutoTokenizer.from_pretrained("defog/sqlcoder-7b-2")
                model = AutoModelForCausalLM.from_pretrained(
                    "defog/sqlcoder-7b-2",
                    torch_dtype=torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config
                )
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
            else:
                raise ValueError(f"Modelo no soportado: {model_name}")
                
            return model, tokenizer
            
        except Exception as e:
            print(f"Error cargando modelo {model_name}: {e}")
            raise
    
    def generate_sql_with_correction(self, model, tokenizer, question: str, 
                                   similar_example: str, max_iterations: int = 3) -> List[Dict]:
        
        attempts = []
        current_sql = None
        error_context = ""
        
        for iteration in range(max_iterations):
            if iteration == 0:
                prompt = f"""You are a SQL expert. Generate ONLY valid SQL for OMOP CDM v5.3.

### Example
{similar_example}

### Question
{question}

### Instructions
- Generate ONLY SQL code, no explanations
- Use standard OMOP table names (PERSON, CONDITION_OCCURRENCE, etc.)
- End with semicolon

SQL:
"""
            else:
                error_brief = error_context[:100] + "..." if len(error_context) > 100 else error_context
                prompt = f"""Fix this SQL error for OMOP CDM v5.3:

### Question
{question}

### Previous SQL (FAILED)
{current_sql[:200]}...

### Error
{error_brief}

Fixed SQL:
"""
            
            inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.05,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
            
            generated_text = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            current_sql = self.clean_generated_sql(generated_text)
            
            validation_error = self.validate_sql_basic(current_sql)
            if validation_error:
                attempt = {
                    'iteration': iteration + 1,
                    'sql': current_sql,
                    'executable': False,
                    'error': f"Validation failed: {validation_error}",
                    'error_type': 'ValidationError',
                    'execution_time': 0,
                }
                attempts.append(attempt)
                error_context = validation_error
                continue
            
            exec_result = self.sql_tester.test_single_query(current_sql)
            
            attempt = {
                'iteration': iteration + 1,
                'sql': current_sql,
                'executable': exec_result['executable'],
                'error': exec_result['error'],
                'error_type': exec_result['error_type'],
                'execution_time': exec_result['execution_time'],
            }
            
            attempts.append(attempt)
            
            if exec_result['executable']:
                break
                
            error_context = exec_result['error'] if exec_result['error'] else "Unknown error"
            
            if len(error_context) > 150:
                error_context = error_context[:150] + "..."
                
        return attempts
    
    def clean_generated_sql(self, generated_text: str) -> str:
        text = generated_text.strip()
        
        lines = text.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if any(phrase in line.lower() for phrase in [
                'explanation:', 'the query', 'this sql', 'note:', 'answer:'
            ]):
                continue
            if line.startswith('--') and len(line) > 50:
                continue
            sql_lines.append(line)
        
        cleaned_sql = '\n'.join(sql_lines).strip()
        
        if cleaned_sql and not cleaned_sql.endswith(';'):
            cleaned_sql += ';'
        
        return cleaned_sql
    
    def validate_sql_basic(self, sql: str) -> Optional[str]:
        if not sql or sql.strip() == ';':
            return "Empty SQL"
        
        sql_upper = sql.upper()
        
        if sql.count('SELECT') > 20:
            return "Too many SELECT statements"
        
        problematic_patterns = [
            'EXPLAIN', 'DESCRIBE', 'SHOW', 'CREATE', 'DROP', 'INSERT', 'UPDATE', 'DELETE'
        ]
        
        for pattern in problematic_patterns:
            if pattern in sql_upper:
                return f"Contains problematic pattern: {pattern}"
        
        return None
    
    def evaluate_single_model(self, model_name: str, test_df: pd.DataFrame) -> pd.DataFrame:
        
        model, tokenizer = self.load_model(model_name)
        
        all_results = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluando"):
            question = row[test_df.columns[test_df.columns.str.upper().str.startswith('QUESTION')][0]]
            ground_truth_sql = row[test_df.columns[test_df.columns.str.upper().str.contains('RUNNABLE')][0]]
            question_id = row.get('ID', idx)
            
            similar_queries = self.retriever.query(question, k=TOP_K_SIMILAR)
            
            if similar_queries:
                score, meta = similar_queries[0]
                similar_example = f"Question: {meta['canonical_question']}\nSQL: {meta['sql']}"
                similar_question = meta['canonical_question']
                similar_sql = meta['sql']
                rag_score = score
            else:
                similar_example = "No similar examples found."
                similar_question = ""
                similar_sql = ""
                rag_score = 0.0
            
            try:
                attempts = self.generate_sql_with_correction(
                    model, tokenizer, question, similar_example, MAX_ITERATIONS
                )
                
                successful_attempts = [a for a in attempts if a['executable']]
                final_attempt = successful_attempts[0] if successful_attempts else attempts[-1]
                
                result = {
                    'question_id': question_id,
                    'model': model_name,
                    'question': question,
                    'reference_sql': ground_truth_sql,
                    'rag_similar_question': similar_question,
                    'rag_similar_sql': similar_sql,
                    'rag_score': rag_score,
                    'final_sql': final_attempt['sql'],
                    'is_executable': final_attempt['executable'],
                    'final_error': final_attempt['error'] if not final_attempt['executable'] else None,
                    'total_attempts': len(attempts),
                    'success_on_attempt': next((i for i, a in enumerate(attempts, 1) if a['executable']), None),
                    'total_execution_time': sum(a['execution_time'] or 0 for a in attempts),
                }
                
                for i, attempt in enumerate(attempts, 1):
                    result[f'attempt_{i}_sql'] = attempt['sql']
                    result[f'attempt_{i}_executable'] = attempt['executable']
                    result[f'attempt_{i}_error'] = attempt['error']
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    'question_id': question_id,
                    'model': model_name,
                    'question': question,
                    'reference_sql': ground_truth_sql,
                    'final_sql': f"ERROR: {str(e)}",
                    'is_executable': False,
                    'final_error': str(e),
                    'total_attempts': 0,
                })
        
        return pd.DataFrame(all_results)
    
    def save_results(self, results_df: pd.DataFrame, model_name: str):
        clean_model_name = model_name.replace("/", "_").replace("-", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.results_dir / f"{clean_model_name}_{timestamp}_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        total_questions = len(results_df)
        executable_count = results_df['is_executable'].sum()
        success_rate = (executable_count / total_questions) * 100 if total_questions > 0 else 0
        
        avg_attempts = results_df['total_attempts'].mean()
        first_attempt_success = (results_df['success_on_attempt'] == 1).sum()
        first_attempt_rate = (first_attempt_success / total_questions) * 100 if total_questions > 0 else 0
        
        report_path = self.results_dir / f"{clean_model_name}_{timestamp}_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"REPORTE DE EVALUACIÓN: {model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total preguntas: {total_questions}\n")
            f.write(f"Consultas ejecutables: {executable_count}\n")
            f.write(f"Tasa de éxito: {success_rate:.1f}%\n")
            f.write(f"Promedio intentos: {avg_attempts:.2f}\n")
            f.write(f"Éxito primer intento: {first_attempt_success} ({first_attempt_rate:.1f}%)\n")
        
        return {
            'total_questions': total_questions,
            'executable_queries': executable_count,
            'success_rate_percent': round(success_rate, 2),
            'first_attempt_success_rate_percent': round(first_attempt_rate, 2),
        }
    
    def run_evaluation(self, models_to_evaluate: List[str]):
        
        if not OMOP_DB_PATH.exists():
            raise FileNotFoundError(f"Base de datos OMOP no encontrada: {OMOP_DB_PATH}")
        
        train_df, test_df = self.split_dataset(DATASET_PATH, TEST_RATIO)
        
        self.retriever = self.build_rag_from_train(train_df)
        
        all_results = []
        
        for model_name in models_to_evaluate:
            try:
                results_df = self.evaluate_single_model(model_name, test_df)
                all_results.append(results_df)
                
                report = self.save_results(results_df, model_name)
                
                print(f"Resumen {model_name}:")
                print(f"   Éxito: {report['executable_queries']}/{report['total_questions']} ({report['success_rate_percent']}%)")
                
            except Exception as e:
                print(f"Error evaluando {model_name}: {e}")
                continue
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            combined_path = self.results_dir / f"combined_evaluation_{timestamp}.csv"
            combined_df.to_csv(combined_path, index=False)

def main():
    models_to_test = [
        "defog/sqlcoder-7b-2",
        "imsanjoykb/sqlCoder-Qwen2.5-8bit",
    ]
    
    try:
        choice = input(f"Selecciona un modelo (1-{len(models_to_test)}) o Enter para el primero: ").strip()
        if choice:
            selected_models = [models_to_test[int(choice) - 1]]
        else:
            selected_models = [models_to_test[0]]
    except (ValueError, IndexError):
        selected_models = [models_to_test[0]]
    
    evaluator = ImprovedSQLEvaluator()
    
    try:
        evaluator.run_evaluation(selected_models)
    except Exception as e:
        print(f"Error durante la evaluación: {e}")

if __name__ == "__main__":
    main()