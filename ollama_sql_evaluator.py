import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import re

from ragTest import MedicalSQLRetriever
from omop_testing.test_sql_execution import SQLExecutabilityTester

DATASET_PATH = Path("text2sql_epi_dataset_omop.xlsx")
OMOP_DB_PATH = Path("omop_testing/omop_test.db")
SCHEMA_PATH = "omop_schema_stub.txt"
TEST_RATIO = 0.2
TOP_K_SIMILAR = 1
MAX_ITERATIONS = 3
RESULTS_DIR = Path("ollama_evaluation_results")
OLLAMA_BASE_URL = "http://localhost:11434"

with open(SCHEMA_PATH, "r") as f:
    db_schema = f.read()

class OllamaModelClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        
    def is_ollama_running(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            return []
        except:
            return []
    
    def generate(self, model_name: str, prompt: str, **kwargs) -> Optional[str]:
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.9),
                    "repeat_penalty": kwargs.get("repeat_penalty", 1.1),
                    "num_predict": kwargs.get("max_tokens", 200),
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("response", "").strip()
            return None
                
        except Exception as e:
            return None

class OllamaSQLEvaluator:
    RECOMMENDED_MODELS = {
        "qwen2.5-coder:32b-instruct-q4_K_M": {
            "name": "Qwen2.5-Coder-32B",
            "temperature": 0.05,
            "max_tokens": 400,
        },
        "deepseek-coder-v2:16b-lite-instruct-q4_K_M": {
            "name": "DeepSeek-Coder-16B",
            "temperature": 0.05,
            "max_tokens": 400,
        },
    }
    
    def __init__(self):
        self.client = OllamaModelClient()
        self.sql_tester = SQLExecutabilityTester(str(OMOP_DB_PATH))
        self.retriever = None
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True)
    
    def create_prompt(self, question: str, similar_example: str, iteration: int = 1, 
                     previous_sql: str = "", error_msg: str = "") -> str:
        
        if iteration > 1:
            error_brief = error_msg[:150] + "..." if len(error_msg) > 150 else error_msg
            prompt = f"""Fix this SQL error for OMOP CDM v5.3:

Question: {question}

Previous SQL (failed):
{previous_sql[:300]}

Error: {error_brief}

Fixed SQL:"""
        else:
            prompt = f"""Generate a SQL query for OMOP CDM v5.3 database.

Question: {question}

Similar example:
{similar_example}

Instructions:
- Use standard OMOP tables (PERSON, CONDITION_OCCURRENCE, DRUG_EXPOSURE, CONCEPT, etc.)
- Generate ONLY SQL code, no explanations
- End with semicolon

SQL:"""
        
        return prompt
    
    def clean_sql_response(self, response: str) -> str:
        if not response:
            return ""
        
        sql_block_patterns = [
            r'```sql\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'SQL:\s*(.*?)(?:\n\n|\Z)',
        ]
        
        for pattern in sql_block_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
                if sql and 'SELECT' in sql.upper():
                    return self.clean_sql_text(sql)
        
        lines = response.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if any(phrase in line.lower() for phrase in [
                'explanation:', 'the query', 'this sql', 'note:',
                'here is', 'this will'
            ]):
                continue
            
            if any(keyword in line.upper() for keyword in [
                'SELECT', 'FROM', 'WHERE', 'JOIN', 'WITH', 'GROUP', 'ORDER', 'HAVING'
            ]):
                sql_lines.append(line)
            elif sql_lines and line and not line.startswith('--'):
                sql_lines.append(line)
        
        if sql_lines:
            return self.clean_sql_text('\n'.join(sql_lines))
        
        return self.clean_sql_text(response)
    
    def clean_sql_text(self, sql: str) -> str:
        lines = sql.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line.startswith('--') and len(line) > 50:
                continue
            if line:
                cleaned_lines.append(line)
        
        sql = '\n'.join(cleaned_lines).strip()
        
        if sql and not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def validate_sql_advanced(self, sql: str) -> Optional[str]:
        if not sql or sql.strip() == ';':
            return "Empty SQL"
        
        sql_upper = sql.upper()
        
        select_count = sql.count('SELECT') + sql.count('select')
        if select_count > 15:
            return "Too many SELECT statements"
        
        required_keywords = ['SELECT', 'FROM']
        if not all(keyword in sql_upper for keyword in required_keywords):
            return "Missing required SQL keywords"
        
        problematic_patterns = [
            'CREATE', 'DROP', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE',
            'EXPLAIN', 'DESCRIBE', 'SHOW'
        ]
        
        for pattern in problematic_patterns:
            if pattern in sql_upper:
                return f"Contains prohibited operation: {pattern}"
        
        if len(sql) > 3000:
            return "SQL too long"
        
        return None
    
    def generate_sql_with_correction(self, model_name: str, question: str, 
                                   similar_example: str, max_iterations: int = 3) -> List[Dict]:
        
        model_config = self.RECOMMENDED_MODELS.get(model_name, {})
        
        attempts = []
        current_sql = None
        error_context = ""
        
        for iteration in range(max_iterations):
            prompt = self.create_prompt(
                question=question,
                similar_example=similar_example,
                iteration=iteration + 1,
                previous_sql=current_sql or "",
                error_msg=error_context
            )
            
            start_time = time.time()
            response = self.client.generate(
                model_name=model_name,
                prompt=prompt,
                temperature=model_config.get("temperature", 0.1),
                max_tokens=model_config.get("max_tokens", 400),
                repeat_penalty=1.1
            )
            generation_time = time.time() - start_time
            
            if not response:
                attempt = {
                    'iteration': iteration + 1,
                    'sql': "ERROR: No response from model",
                    'executable': False,
                    'error': "Model generation failed",
                    'error_type': 'GenerationError',
                    'execution_time': None,
                    'generation_time': generation_time,
                }
                attempts.append(attempt)
                break
            
            current_sql = self.clean_sql_response(response)
            
            validation_error = self.validate_sql_advanced(current_sql)
            if validation_error:
                attempt = {
                    'iteration': iteration + 1,
                    'sql': current_sql,
                    'executable': False,
                    'error': f"Validation: {validation_error}",
                    'error_type': 'ValidationError',
                    'execution_time': None,
                    'generation_time': generation_time,
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
                'generation_time': generation_time,
            }
            
            attempts.append(attempt)
            
            if exec_result['executable']:
                break
            
            error_context = exec_result['error'] if exec_result['error'] else "Unknown error"
            if len(error_context) > 200:
                error_context = error_context[:200] + "..."
                
        return attempts
    
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
    
    def evaluate_model(self, model_name: str, test_df: pd.DataFrame) -> pd.DataFrame:
        
        model_config = self.RECOMMENDED_MODELS.get(model_name, {})
        model_display_name = model_config.get("name", model_name)
        
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
                    model_name, question, similar_example, MAX_ITERATIONS
                )
                
                successful_attempts = [a for a in attempts if a['executable']]
                final_attempt = successful_attempts[0] if successful_attempts else attempts[-1]
                
                result = {
                    'question_id': question_id,
                    'model': model_display_name,
                    'model_full_name': model_name,
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
                    'total_generation_time': sum(a.get('generation_time', 0) for a in attempts),
                    'total_execution_time': sum(a.get('execution_time', 0) or 0 for a in attempts),
                }
                
                for i, attempt in enumerate(attempts, 1):
                    result[f'attempt_{i}_sql'] = attempt['sql']
                    result[f'attempt_{i}_executable'] = attempt['executable']
                    result[f'attempt_{i}_error'] = attempt['error']
                    result[f'attempt_{i}_generation_time'] = attempt.get('generation_time', 0)
                
                all_results.append(result)
                
            except Exception as e:
                all_results.append({
                    'question_id': question_id,
                    'model': model_display_name,
                    'model_full_name': model_name,
                    'question': question,
                    'reference_sql': ground_truth_sql,
                    'final_sql': f"ERROR: {str(e)}",
                    'is_executable': False,
                    'final_error': str(e),
                    'total_attempts': 0,
                })
        
        return pd.DataFrame(all_results)
    
    def save_results(self, results_df: pd.DataFrame, model_name: str):
        clean_model_name = model_name.replace("/", "_").replace(":", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        csv_path = self.results_dir / f"ollama_{clean_model_name}_{timestamp}_results.csv"
        results_df.to_csv(csv_path, index=False)
        
        total = len(results_df)
        executable = results_df['is_executable'].sum()
        success_rate = (executable / total) * 100 if total > 0 else 0
        
        first_attempt_success = (results_df['success_on_attempt'] == 1).sum()
        first_attempt_rate = (first_attempt_success / total) * 100 if total > 0 else 0
        
        avg_attempts = results_df['total_attempts'].mean()
        avg_gen_time = results_df['total_generation_time'].mean()
        
        report_path = self.results_dir / f"ollama_{clean_model_name}_{timestamp}_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"EVALUACIÓN OLLAMA: {model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Fecha: {timestamp}\n")
            f.write(f"Total consultas: {total}\n")
            f.write(f"Consultas ejecutables: {executable}\n")
            f.write(f"Tasa de éxito: {success_rate:.1f}%\n")
            f.write(f"Éxito primer intento: {first_attempt_success} ({first_attempt_rate:.1f}%)\n")
            f.write(f"Promedio intentos: {avg_attempts:.2f}\n")
            f.write(f"Tiempo promedio generación: {avg_gen_time:.2f}s\n\n")
            
            error_counts = results_df[results_df['is_executable'] == False]['final_error'].value_counts()
            f.write("ERRORES MÁS COMUNES:\n")
            f.write("-" * 30 + "\n")
            for error, count in error_counts.head(5).items():
                f.write(f"{count:2d}x {error}\n")
        
        return csv_path
    
    def run_evaluation(self, models_to_test: List[str]):
        
        if not self.client.is_ollama_running():
            print("Ollama no está ejecutándose. Ejecuta: ollama serve")
            return
        
        if not OMOP_DB_PATH.exists():
            raise FileNotFoundError(f"Base de datos OMOP no encontrada: {OMOP_DB_PATH}")
        
        train_df, test_df = self.split_dataset(DATASET_PATH, TEST_RATIO)
        
        self.retriever = self.build_rag_from_train(train_df)
        
        all_results = []
        for model_name in models_to_test:
            try:
                results_df = self.evaluate_model(model_name, test_df)
                if not results_df.empty:
                    all_results.append(results_df)
                    self.save_results(results_df, model_name)
                    
                    executable = results_df['is_executable'].sum()
                    total = len(results_df)
                    success_rate = (executable / total) * 100
                    first_attempt = (results_df['success_on_attempt'] == 1).sum()
                    
                    print(f"Resumen {model_name}:")
                    print(f"   Éxito: {executable}/{total} ({success_rate:.1f}%)")
                    print(f"   Primer intento: {first_attempt}")
                    
            except Exception as e:
                print(f"Error evaluando {model_name}: {e}")

def main():
    evaluator = OllamaSQLEvaluator()
    
    available_models = evaluator.client.list_models()
    recommended = list(evaluator.RECOMMENDED_MODELS.keys())
    
    selected_models = []
    for model in recommended:
        if any(model in m for m in available_models):
            selected_models.append(model)
    
    if not selected_models:
        print("No hay modelos recomendados disponibles")
        return
    
    evaluator.run_evaluation(selected_models)

if __name__ == "__main__":
    main()