import sqlite3
from pathlib import Path

REQUIRED_TABLES = [
    'CONCEPT', 'PERSON', 'CONDITION_OCCURRENCE', 'DRUG_EXPOSURE', 
    'CONDITION_ERA', 'PROCEDURE_OCCURRENCE', 'OBSERVATION', 
    'CONCEPT_ANCESTOR', 'DRUG_ERA', 'VISIT_OCCURRENCE', 'DEATH', 
    'CONCEPT_RELATIONSHIP', 'PROVIDER'
]

OMOP_DDL_SQLITE = {
    'CONCEPT': """
        CREATE TABLE concept (
            concept_id INTEGER NOT NULL,
            concept_name TEXT NOT NULL,
            domain_id TEXT NOT NULL,
            vocabulary_id TEXT NOT NULL,
            concept_class_id TEXT NOT NULL,
            standard_concept TEXT NULL,
            concept_code TEXT NOT NULL,
            valid_start_date TEXT NOT NULL,
            valid_end_date TEXT NOT NULL,
            invalid_reason TEXT NULL,
            PRIMARY KEY (concept_id)
        );
    """,
    
    'PERSON': """
        CREATE TABLE person (
            person_id INTEGER NOT NULL,
            gender_concept_id INTEGER NOT NULL,
            year_of_birth INTEGER NOT NULL,
            month_of_birth INTEGER NULL,
            day_of_birth INTEGER NULL,
            birth_datetime TEXT NULL,
            race_concept_id INTEGER NOT NULL,
            ethnicity_concept_id INTEGER NOT NULL,
            location_id INTEGER NULL,
            provider_id INTEGER NULL,
            care_site_id INTEGER NULL,
            person_source_value TEXT NULL,
            gender_source_value TEXT NULL,
            gender_source_concept_id INTEGER NULL,
            race_source_value TEXT NULL,
            race_source_concept_id INTEGER NULL,
            ethnicity_source_value TEXT NULL,
            ethnicity_source_concept_id INTEGER NULL,
            PRIMARY KEY (person_id)
        );
    """,
    
    'CONDITION_OCCURRENCE': """
        CREATE TABLE condition_occurrence (
            condition_occurrence_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            condition_concept_id INTEGER NOT NULL,
            condition_start_date TEXT NOT NULL,
            condition_start_datetime TEXT NULL,
            condition_end_date TEXT NULL,
            condition_end_datetime TEXT NULL,
            condition_type_concept_id INTEGER NOT NULL,
            condition_status_concept_id INTEGER NULL,
            stop_reason TEXT NULL,
            provider_id INTEGER NULL,
            visit_occurrence_id INTEGER NULL,
            visit_detail_id INTEGER NULL,
            condition_source_value TEXT NULL,
            condition_source_concept_id INTEGER NULL,
            condition_status_source_value TEXT NULL,
            PRIMARY KEY (condition_occurrence_id)
        );
    """,
    
    'DRUG_EXPOSURE': """
        CREATE TABLE drug_exposure (
            drug_exposure_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            drug_concept_id INTEGER NOT NULL,
            drug_exposure_start_date TEXT NOT NULL,
            drug_exposure_start_datetime TEXT NULL,
            drug_exposure_end_date TEXT NOT NULL,
            drug_exposure_end_datetime TEXT NULL,
            verbatim_end_date TEXT NULL,
            drug_type_concept_id INTEGER NOT NULL,
            stop_reason TEXT NULL,
            refills INTEGER NULL,
            quantity REAL NULL,
            days_supply INTEGER NULL,
            sig TEXT NULL,
            route_concept_id INTEGER NULL,
            lot_number TEXT NULL,
            provider_id INTEGER NULL,
            visit_occurrence_id INTEGER NULL,
            visit_detail_id INTEGER NULL,
            drug_source_value TEXT NULL,
            drug_source_concept_id INTEGER NULL,
            route_source_value TEXT NULL,
            dose_unit_source_value TEXT NULL,
            PRIMARY KEY (drug_exposure_id)
        );
    """,
    
    'CONDITION_ERA': """
        CREATE TABLE condition_era (
            condition_era_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            condition_concept_id INTEGER NOT NULL,
            condition_era_start_date TEXT NOT NULL,
            condition_era_end_date TEXT NOT NULL,
            condition_occurrence_count INTEGER NULL,
            PRIMARY KEY (condition_era_id)
        );
    """,
    
    'PROCEDURE_OCCURRENCE': """
        CREATE TABLE procedure_occurrence (
            procedure_occurrence_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            procedure_concept_id INTEGER NOT NULL,
            procedure_date TEXT NOT NULL,
            procedure_datetime TEXT NULL,
            procedure_type_concept_id INTEGER NOT NULL,
            modifier_concept_id INTEGER NULL,
            quantity INTEGER NULL,
            provider_id INTEGER NULL,
            visit_occurrence_id INTEGER NULL,
            visit_detail_id INTEGER NULL,
            procedure_source_value TEXT NULL,
            procedure_source_concept_id INTEGER NULL,
            modifier_source_value TEXT NULL,
            PRIMARY KEY (procedure_occurrence_id)
        );
    """,
    
    'OBSERVATION': """
        CREATE TABLE observation (
            observation_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            observation_concept_id INTEGER NOT NULL,
            observation_date TEXT NOT NULL,
            observation_datetime TEXT NULL,
            observation_type_concept_id INTEGER NOT NULL,
            value_as_number REAL NULL,
            value_as_string TEXT NULL,
            value_as_concept_id INTEGER NULL,
            qualifier_concept_id INTEGER NULL,
            unit_concept_id INTEGER NULL,
            provider_id INTEGER NULL,
            visit_occurrence_id INTEGER NULL,
            visit_detail_id INTEGER NULL,
            observation_source_value TEXT NULL,
            observation_source_concept_id INTEGER NULL,
            unit_source_value TEXT NULL,
            qualifier_source_value TEXT NULL,
            PRIMARY KEY (observation_id)
        );
    """,
    
    'CONCEPT_ANCESTOR': """
        CREATE TABLE concept_ancestor (
            ancestor_concept_id INTEGER NOT NULL,
            descendant_concept_id INTEGER NOT NULL,
            min_levels_of_separation INTEGER NOT NULL,
            max_levels_of_separation INTEGER NOT NULL
        );
    """,
    
    'DRUG_ERA': """
        CREATE TABLE drug_era (
            drug_era_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            drug_concept_id INTEGER NOT NULL,
            drug_era_start_date TEXT NOT NULL,
            drug_era_end_date TEXT NOT NULL,
            drug_exposure_count INTEGER NULL,
            gap_days INTEGER NULL,
            PRIMARY KEY (drug_era_id)
        );
    """,
    
    'VISIT_OCCURRENCE': """
        CREATE TABLE visit_occurrence (
            visit_occurrence_id INTEGER NOT NULL,
            person_id INTEGER NOT NULL,
            visit_concept_id INTEGER NOT NULL,
            visit_start_date TEXT NOT NULL,
            visit_start_datetime TEXT NULL,
            visit_end_date TEXT NOT NULL,
            visit_end_datetime TEXT NULL,
            visit_type_concept_id INTEGER NOT NULL,
            provider_id INTEGER NULL,
            care_site_id INTEGER NULL,
            visit_source_value TEXT NULL,
            visit_source_concept_id INTEGER NULL,
            admitted_from_concept_id INTEGER NULL,
            admitted_from_source_value TEXT NULL,
            discharge_to_concept_id INTEGER NULL,
            discharge_to_source_value TEXT NULL,
            preceding_visit_occurrence_id INTEGER NULL,
            PRIMARY KEY (visit_occurrence_id)
        );
    """,
    
    'DEATH': """
        CREATE TABLE death (
            person_id INTEGER NOT NULL,
            death_date TEXT NOT NULL,
            death_datetime TEXT NULL,
            death_type_concept_id INTEGER NOT NULL,
            cause_concept_id INTEGER NULL,
            cause_source_value TEXT NULL,
            cause_source_concept_id INTEGER NULL,
            PRIMARY KEY (person_id)
        );
    """,
    
    'CONCEPT_RELATIONSHIP': """
        CREATE TABLE concept_relationship (
            concept_id_1 INTEGER NOT NULL,
            concept_id_2 INTEGER NOT NULL,
            relationship_id TEXT NOT NULL,
            valid_start_date TEXT NOT NULL,
            valid_end_date TEXT NOT NULL,
            invalid_reason TEXT NULL
        );
    """,
    
    'PROVIDER': """
        CREATE TABLE provider (
            provider_id INTEGER NOT NULL,
            provider_name TEXT NULL,
            npi TEXT NULL,
            dea TEXT NULL,
            specialty_concept_id INTEGER NULL,
            care_site_id INTEGER NULL,
            year_of_birth INTEGER NULL,
            gender_concept_id INTEGER NULL,
            provider_source_value TEXT NULL,
            specialty_source_value TEXT NULL,
            specialty_source_concept_id INTEGER NULL,
            gender_source_value TEXT NULL,
            gender_source_concept_id INTEGER NULL,
            PRIMARY KEY (provider_id)
        );
    """
}

def create_omop_database(db_path: str = "omop_testing/omop_test.db"):
    
    Path("omop_testing").mkdir(exist_ok=True)
    db_path = Path(db_path)
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            for table in REQUIRED_TABLES:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
            for table in REQUIRED_TABLES:
                if table in OMOP_DDL_SQLITE:
                    cursor.execute(OMOP_DDL_SQLITE[table])
            
            conn.commit()
            print(f"Base de datos OMOP creada: {db_path}")
            
            # Test simple
            cursor.execute("SELECT COUNT(*) FROM person")
            print("Base de datos funcionando correctamente")
            
            return True
            
    except Exception as e:
        print(f"Error creando la base de datos: {e}")
        return False

def main():
    create_omop_database()

if __name__ == "__main__":
    main()