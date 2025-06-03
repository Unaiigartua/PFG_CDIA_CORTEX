import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import time
import json

class SQLExecutabilityTester:
    
    def __init__(self, db_path: str = "omop_testing/omop_test.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Base de datos no encontrada: {self.db_path}")
            
    def test_single_query(self, sql: str, timeout: int = 30) -> Dict[str, any]:
        result = {
            'sql': sql,
            'executable': False,
            'error': None,
            'execution_time': None,
            'row_count': None,
            'error_type': None
        }
        
        try:
            start_time = time.time()
            
            with sqlite3.connect(self.db_path, timeout=timeout) as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                
                if sql.strip().upper().startswith('SELECT'):
                    rows = cursor.fetchall()
                    result['row_count'] = len(rows)
                else:
                    result['row_count'] = cursor.rowcount
                    
                result['executable'] = True
                result['execution_time'] = time.time() - start_time
                
        except sqlite3.OperationalError as e:
            result['error'] = str(e)
            result['error_type'] = 'OperationalError'
        except sqlite3.DatabaseError as e:
            result['error'] = str(e)
            result['error_type'] = 'DatabaseError'
        except Exception as e:
            result['error'] = str(e)
            result['error_type'] = type(e).__name__
            
        return result
    
    def test_queries_batch(self, queries: List[str]) -> List[Dict[str, any]]:
        results = []
        
        for sql in queries:
            result = self.test_single_query(sql)
            results.append(result)
                
        return results
    
    def test_from_excel(self, excel_path: str, sql_column: str = "QUERY_SNOWFLAKE_RUNNABLE") -> pd.DataFrame:
        df = pd.read_excel(excel_path)
        
        if sql_column not in df.columns:
            raise ValueError(f"Columna '{sql_column}' no encontrada en el Excel")
            
        queries = df[sql_column].dropna().tolist()
        
        results = self.test_queries_batch(queries)
        
        results_df = pd.DataFrame(results)
        
        for i, result in enumerate(results):
            if i < len(df):
                results_df.loc[i, 'original_id'] = df.iloc[i].get('ID', i+1)
                results_df.loc[i, 'question'] = df.iloc[i].get('QUESTION', '')
                
        return results_df
    
    def generate_report(self, results: List[Dict[str, any]]) -> Dict[str, any]:
        total = len(results)
        executable = sum(1 for r in results if r['executable'])
        failed = total - executable
        
        error_types = {}
        for r in results:
            if not r['executable'] and r['error_type']:
                error_types[r['error_type']] = error_types.get(r['error_type'], 0) + 1
                
        exec_times = [r['execution_time'] for r in results if r['execution_time'] is not None]
        avg_time = sum(exec_times) / len(exec_times) if exec_times else 0
        
        report = {
            'total_queries': total,
            'executable': executable,
            'failed': failed,
            'success_rate': (executable / total * 100) if total > 0 else 0,
            'error_types': error_types,
            'average_execution_time': avg_time,
            'total_execution_time': sum(exec_times)
        }
        
        return report
    
    def save_results(self, results: List[Dict[str, any]], output_path: str):
        output_path = Path(output_path)
        
        df = pd.DataFrame(results)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        json_path = output_path.with_suffix('.json')
        report = self.generate_report(results)
        
        output_data = {
            'report': report,
            'results': results
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    try:
        tester = SQLExecutabilityTester()
        
        excel_path = "text2sql_epi_dataset_omop.xlsx"
        if Path(excel_path).exists():
            results_df = tester.test_from_excel(excel_path)
            
            output_path = Path("omop_testing") / "sql_executability_results"
            tester.save_results(results_df.to_dict('records'), output_path)
            
            report = tester.generate_report(results_df.to_dict('records'))
            print(f"Total consultas: {report['total_queries']}")
            print(f"Ejecutables: {report['executable']} ({report['success_rate']:.1f}%)")
            print(f"Con errores: {report['failed']}")
            
        else:
            test_sql = """
            SELECT COUNT(DISTINCT p.person_id) AS num_patients
            FROM PERSON p
            JOIN CONDITION_OCCURRENCE co ON p.person_id = co.person_id
            WHERE co.condition_concept_id IN (133834, 4298597)
            """
            
            result = tester.test_single_query(test_sql)
            
            if result['executable']:
                print("Consulta ejecutable")
                print(f"Tiempo: {result['execution_time']:.3f}s")
            else:
                print(f"Error: {result['error']}")
                
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Ejecuta create_omop_sqlite.py primero")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()