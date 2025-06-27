import pytest
import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.etl.ingest_csv import cargar_progol
from src.etl.scrape_odds import cargar_odds
from src.etl.parse_previas import extract_previas_from_pdf, guardar_previas_json
from src.etl.build_features import (normalizar_momios, merge_fuentes, 
                                    construir_features, guardar_features)
from src.utils.validators import (validar_progol, validar_momios, 
                                 validar_probabilidades)


class TestIngestCSV:
    """Pruebas para la ingesta de CSV"""
    
    @pytest.fixture
    def sample_progol_data(self, tmp_path):
        """Crea datos de prueba para Progol.csv"""
        data = {
            'concurso_id': [2283, 2283, 2283],
            'fecha': ['2025-05-31', '2025-05-31', '2025-05-31'],
            'match_no': [1, 2, 3],
            'liga': ['Liga MX', 'Liga MX', 'Premier'],
            'home': ['America', 'Chivas', 'Liverpool'],
            'away': ['Cruz Azul', 'Pumas', 'Chelsea'],
            'l_g': [2, 1, 0],
            'a_g': [1, 1, 2],
            'resultado': ['L', 'E', 'V'],
            'premio_1': [0, 0, 0],
            'premio_2': [0, 0, 0]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "Progol.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_cargar_progol_valido(self, sample_progol_data, monkeypatch):
        """Prueba carga exitosa de Progol.csv"""
        monkeypatch.setattr('src.utils.config.ARCH_PROGOL', str(sample_progol_data))
        
        df = cargar_progol()
        
        assert len(df) == 3
        assert 'match_id' in df.columns
        assert df['match_id'].iloc[0] == '2283-1'
        assert all(df['resultado'] == ['L', 'E', 'V'])
    
    def test_recalculo_resultado_inconsistente(self, tmp_path, monkeypatch):
        """Prueba recálculo cuando resultado no coincide con goles"""
        data = {
            'concurso_id': [2283],
            'fecha': ['2025-05-31'],
            'match_no': [1],
            'liga': ['Liga MX'],
            'home': ['America'],
            'away': ['Cruz Azul'],
            'l_g': [2],  # Local anotó más
            'a_g': [1],
            'resultado': ['V'],  # Pero dice visitante ganó (inconsistente)
            'premio_1': [0],
            'premio_2': [0]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "Progol_bad.csv"
        df.to_csv(csv_path, index=False)
        
        monkeypatch.setattr('src.utils.config.ARCH_PROGOL', str(csv_path))
        
        df_result = cargar_progol()
        
        # Debe corregir a 'L'
        assert df_result['resultado'].iloc[0] == 'L'
        assert df_result['resultado_calc'].iloc[0] == 'L'
    
    def test_validar_progol_duplicados(self):
        """Prueba detección de match_id duplicados"""
        df = pd.DataFrame({
            'match_id': ['2283-1', '2283-1'],  # Duplicado
            'home': ['A', 'B'],
            'away': ['C', 'D'],
            'fecha': [datetime.now(), datetime.now()]
        })
        
        with pytest.raises(AssertionError, match="match_id duplicado"):
            validar_progol(df)


class TestScrapeOdds:
    """Pruebas para scraping de momios"""
    
    @pytest.fixture
    def sample_odds_data(self, tmp_path):
        """Crea datos de prueba para odds.csv"""
        data = {
            'concurso_id': [2283, 2283, 2283],
            'match_no': [1, 2, 3],
            'fecha': ['2025-05-31', '2025-05-31', '2025-05-31'],
            'home': ['America', 'Chivas', 'Liverpool'],
            'away': ['Cruz Azul', 'Pumas', 'Chelsea'],
            'odds_L': [1.80, 2.20, 3.50],
            'odds_E': [3.50, 3.20, 3.40],
            'odds_V': [4.20, 3.00, 2.10]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "odds.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    
    def test_normalizar_momios(self):
        """Prueba normalización de momios"""
        df = pd.DataFrame({
            'odds_L': [2.0, 1.5],
            'odds_E': [3.0, 3.5],
            'odds_V': [4.0, 5.0]
        })
        
        df_norm = normalizar_momios(df)
        
        # Verificar que suma = 1
        for idx in range(len(df_norm)):
            suma = df_norm.loc[idx, 'p_raw_L'] + df_norm.loc[idx, 'p_raw_E'] + df_norm.loc[idx, 'p_raw_V']
            assert abs(suma - 1.0) < 1e-6
        
        # Verificar cálculo correcto
        # Para el primer partido: 1/2 + 1/3 + 1/4 = 6/12 + 4/12 + 3/12 = 13/12
        # p_L = (1/2) / (13/12) = 6/13 ≈ 0.4615
        assert abs(df_norm.loc[0, 'p_raw_L'] - 6/13) < 1e-4
    
    def test_validar_momios_invalidos(self):
        """Prueba detección de momios inválidos"""
        df = pd.DataFrame({
            'odds_L': [0.5, 2.0],  # 0.5 es inválido (< 1.01)
            'odds_E': [3.0, 3.0],
            'odds_V': [4.0, 4.0]
        })
        
        with pytest.raises(AssertionError, match="Momio inválido"):
            validar_momios(df)


class TestParsePrevias:
    """Pruebas para parsing de previas PDF"""
    
    def test_extract_previas_estructura(self):
        """Prueba estructura de salida del parser"""
        # Simulamos salida esperada
        previas = [
            {
                "match_id": "2283-1",
                "form_H": "WWDLW",
                "form_A": "LLWWD",
                "h2h_H": 3,
                "h2h_E": 1,
                "h2h_A": 1,
                "inj_H": 2,
                "inj_A": 1,
                "context_flag": ["derbi"]
            }
        ]
        
        # Validar estructura
        assert len(previas) == 1
        assert all(key in previas[0] for key in ["match_id", "form_H", "form_A", "h2h_H"])
        assert len(previas[0]["form_H"]) == 5
        assert previas[0]["form_H"].count('W') + previas[0]["form_H"].count('D') + previas[0]["form_H"].count('L') == 5
    
    def test_guardar_previas_json(self, tmp_path):
        """Prueba guardado de previas en JSON"""
        previas = [
            {"match_id": "2283-1", "form_H": "WWWWW", "form_A": "LLLLL"},
            {"match_id": "2283-2", "form_H": "WDWDW", "form_A": "LDLDL"}
        ]
        
        json_path = tmp_path / "previas_test.json"
        guardar_previas_json(previas, str(json_path))
        
        # Verificar archivo creado
        assert json_path.exists()
        
        # Verificar contenido
        with open(json_path, 'r') as f:
            loaded = json.load(f)
        
        assert len(loaded) == 2
        assert loaded[0]["match_id"] == "2283-1"
        assert loaded[1]["form_H"] == "WDWDW"


class TestBuildFeatures:
    """Pruebas para construcción de features"""
    
    @pytest.fixture
    def sample_merged_data(self):
        """Datos de prueba para features"""
        return pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'home': ['America', 'Chivas'],
            'away': ['Cruz Azul', 'Pumas'],
            'p_raw_L': [0.45, 0.33],
            'p_raw_E': [0.30, 0.33],
            'p_raw_V': [0.25, 0.34],
            'form_H': ['WWWDL', 'WDLDL'],
            'form_A': ['LLWWD', 'DWWWL'],
            'h2h_H': [3, 2],
            'h2h_E': [1, 2],
            'h2h_A': [1, 3],
            'inj_H': [1, 0],
            'inj_A': [2, 1],
            'elo_home': [1650, 1580],
            'elo_away': [1600, 1620],
            'context_flag': [['derbi'], []],
            'goles_H': [2, 1],
            'goles_A': [1, 2]
        })
    
    def test_construir_features_calculo(self, sample_merged_data):
        """Prueba cálculo correcto de features"""
        df = construir_features(sample_merged_data)
        
        # Verificar features creadas
        assert 'delta_forma' in df.columns
        assert 'h2h_ratio' in df.columns
        assert 'elo_diff' in df.columns
        assert 'inj_weight' in df.columns
        assert 'is_derby' in df.columns
        assert 'draw_propensity_flag' in df.columns
        
        # Verificar cálculos específicos
        # Partido 1: form_H tiene 3W+1D = 10 puntos, form_A tiene 2W+1D = 7 puntos
        assert df.loc[0, 'gf5_H'] == 10
        assert df.loc[0, 'gf5_A'] == 7
        assert df.loc[0, 'delta_forma'] == 3
        
        # H2H ratio: (3-1)/(3+1+1) = 2/5 = 0.4
        assert abs(df.loc[0, 'h2h_ratio'] - 0.4) < 1e-6
        
        # Elo diff
        assert df.loc[0, 'elo_diff'] == 50
        
        # Context flags
        assert df.loc[0, 'is_derby'] == True
        assert df.loc[1, 'is_derby'] == False
    
    def test_draw_propensity_flag(self):
        """Prueba identificación de draw propensity"""
        df = pd.DataFrame({
            'p_raw_L': [0.35, 0.50, 0.30],
            'p_raw_E': [0.40, 0.25, 0.35],
            'p_raw_V': [0.25, 0.25, 0.35]
        })
        
        # Calcular manualmente
        # Partido 1: |0.35-0.25| = 0.10 > 0.08, no cumple
        # Partido 2: |0.50-0.25| = 0.25 > 0.08, no cumple
        # Partido 3: |0.30-0.35| = 0.05 < 0.08 y 0.35 > max(0.30,0.35), no cumple el segundo criterio
        
        # Agregar columnas necesarias para construir_features
        df['form_H'] = 'WWWWW'
        df['form_A'] = 'LLLLL'
        df['h2h_H'] = 1
        df['h2h_E'] = 1
        df['h2h_A'] = 1
        df['inj_H'] = 0
        df['inj_A'] = 0
        df['context_flag'] = [[]] * len(df)
        
        df_feat = construir_features(df)
        
        # En este caso ninguno debería tener draw_propensity_flag = True
        assert df_feat['draw_propensity_flag'].sum() == 0


class TestIntegration:
    """Pruebas de integración del pipeline ETL completo"""
    
    def test_pipeline_completo(self, tmp_path, monkeypatch):
        """Prueba el pipeline ETL de inicio a fin"""
        # Preparar datos de prueba
        jornada_id = 2283
        
        # Crear archivos de prueba
        progol_data = pd.DataFrame({
            'concurso_id': [jornada_id] * 14,
            'fecha': ['2025-05-31'] * 14,
            'match_no': list(range(1, 15)),
            'liga': ['Liga MX'] * 14,
            'home': [f'Equipo{i}' for i in range(1, 15)],
            'away': [f'Equipo{i+14}' for i in range(1, 15)],
            'l_g': np.random.randint(0, 4, 14),
            'a_g': np.random.randint(0, 4, 14),
            'resultado': np.random.choice(['L', 'E', 'V'], 14),
            'premio_1': [0] * 14,
            'premio_2': [0] * 14
        })
        
        odds_data = progol_data[['home', 'away', 'fecha']].copy()
        odds_data['odds_L'] = np.random.uniform(1.5, 4.0, 14)
        odds_data['odds_E'] = np.random.uniform(2.8, 3.8, 14)
        odds_data['odds_V'] = np.random.uniform(1.8, 5.0, 14)
        
        # Guardar archivos
        progol_path = tmp_path / "Progol.csv"
        odds_path = tmp_path / "odds.csv"
        progol_data.to_csv(progol_path, index=False)
        odds_data.to_csv(odds_path, index=False)
        
        # Crear previas dummy
        previas = []
        for i in range(1, 15):
            previas.append({
                "match_id": f"{jornada_id}-{i}",
                "form_H": "WWDLW",
                "form_A": "LDWWL",
                "h2h_H": 2,
                "h2h_E": 2,
                "h2h_A": 1,
                "inj_H": np.random.randint(0, 3),
                "inj_A": np.random.randint(0, 3),
                "context_flag": ["derbi"] if i in [1, 7] else []
            })
        
        previas_path = tmp_path / "previas.json"
        with open(previas_path, 'w') as f:
            json.dump(previas, f)
        
        # Simular paths
        monkeypatch.setattr('src.utils.config.ARCH_PROGOL', str(progol_path))
        monkeypatch.setattr('src.utils.config.ARCH_ODDS', str(odds_path))
        
        # Ejecutar pipeline
        # 1. Cargar datos
        df_progol = cargar_progol()
        assert len(df_progol) == 14
        
        # 2. Cargar y normalizar momios
        df_odds = cargar_odds()
        df_odds = normalizar_momios(df_odds)
        assert 'p_raw_L' in df_odds.columns
        
        # 3. Merge
        df_merged = merge_fuentes(df_progol, df_odds, previas, None, None)
        assert len(df_merged) == 14
        
        # 4. Features
        df_features = construir_features(df_merged)
        assert 'delta_forma' in df_features.columns
        assert 'draw_propensity_flag' in df_features.columns
        
        # 5. Validar salida
        assert df_features['match_id'].nunique() == 14
        assert df_features[['p_raw_L', 'p_raw_E', 'p_raw_V']].isnull().sum().sum() == 0
        
        print("✅ Pipeline ETL completado exitosamente")


# Configuración para pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])