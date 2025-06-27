import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.modeling.poisson_model import (preparar_dataset_poisson, entrenar_poisson_model,
                                       predecir_lambdas, guardar_lambda)
from src.modeling.stacking import bivariate_poisson_probs, stack_probabilities
from src.modeling.bayesian_adjustment import ajustar_bayes
from src.modeling.draw_propensity import aplicar_draw_propensity
from src.utils.validators import validar_probabilidades


class TestPoissonModel:
    """Pruebas para el modelo Poisson"""
    
    @pytest.fixture
    def sample_match_data(self):
        """Datos de ejemplo para entrenar modelo"""
        return pd.DataFrame({
            'match_id': ['2283-1', '2283-2', '2283-3'],
            'home': ['America', 'Chivas', 'Pumas'],
            'away': ['Cruz Azul', 'Atlas', 'Tigres'],
            'liga': ['Liga MX', 'Liga MX', 'Liga MX'],
            'goles_H': [2, 1, 3],
            'goles_A': [1, 1, 0],
            'elo_home': [1650, 1600, 1700],
            'elo_away': [1600, 1580, 1620],
            'factor_local': [0.45, 0.45, 0.45]
        })
    
    def test_preparar_dataset_poisson(self, sample_match_data):
        """Prueba preparación de datos para Poisson"""
        df = preparar_dataset_poisson(sample_match_data)
        
        assert 'elo_diff' in df.columns
        assert 'log_lambda_1' in df.columns
        assert 'log_lambda_2' in df.columns
        
        # Verificar cálculos
        assert df.loc[0, 'elo_diff'] == 50
        assert abs(df.loc[0, 'log_lambda_1'] - np.log1p(2)) < 1e-6
        assert abs(df.loc[1, 'log_lambda_2'] - np.log1p(1)) < 1e-6
    
    @patch('sklearn.linear_model.Ridge')
    def test_entrenar_poisson_model(self, mock_ridge, sample_match_data):
        """Prueba entrenamiento del modelo"""
        # Mock de Ridge
        mock_model = Mock()
        mock_model.fit = Mock()
        mock_ridge.return_value = mock_model
        
        df = preparar_dataset_poisson(sample_match_data)
        model_H, model_A, ohe = entrenar_poisson_model(df)
        
        # Verificar que se crearon los modelos
        assert model_H is not None
        assert model_A is not None
        assert ohe is not None
        
        # Verificar que se llamó fit
        assert mock_model.fit.called
    
    def test_predecir_lambdas_validos(self, sample_match_data):
        """Prueba que lambdas predichos sean válidos"""
        # Crear modelos mock que devuelven valores conocidos
        mock_model_H = Mock()
        mock_model_H.predict = Mock(return_value=np.array([1.5, 1.2, 2.0]))
        
        mock_model_A = Mock()
        mock_model_A.predict = Mock(return_value=np.array([1.0, 1.1, 0.8]))
        
        # Mock OneHotEncoder
        mock_ohe = Mock()
        mock_ohe.transform = Mock(return_value=np.zeros((3, 10)))
        mock_ohe.get_feature_names_out = Mock(return_value=[f'feature_{i}' for i in range(10)])
        
        df = sample_match_data.copy()
        df_lambda = predecir_lambdas(df, mock_model_H, mock_model_A, mock_ohe)
        
        # Verificar estructura
        assert len(df_lambda) == 3
        assert 'lambda1' in df_lambda.columns
        assert 'lambda2' in df_lambda.columns
        assert 'lambda3' in df_lambda.columns
        
        # Verificar valores válidos
        assert (df_lambda['lambda1'] > 0).all()
        assert (df_lambda['lambda2'] > 0).all()
        assert (df_lambda['lambda3'] >= 0).all()
        assert (df_lambda['lambda3'] <= 1).all()


class TestStacking:
    """Pruebas para stacking de probabilidades"""
    
    def test_bivariate_poisson_probs(self):
        """Prueba cálculo de probabilidades bivariadas"""
        # Caso simple con lambdas conocidos
        lambda1 = np.array([1.5])
        lambda2 = np.array([1.0])
        lambda3 = np.array([0.1])
        
        p_L, p_E, p_V = bivariate_poisson_probs(lambda1, lambda2, lambda3, max_goals=5)
        
        # Verificar que suman a 1
        assert abs(p_L[0] + p_E[0] + p_V[0] - 1.0) < 1e-6
        
        # Con lambda1 > lambda2, p_L debería ser mayor
        assert p_L[0] > p_V[0]
    
    def test_stack_probabilities(self):
        """Prueba combinación de probabilidades"""
        df_features = pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'p_raw_L': [0.45, 0.30],
            'p_raw_E': [0.30, 0.35],
            'p_raw_V': [0.25, 0.35]
        })
        
        df_lambda = pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'lambda1': [1.5, 1.0],
            'lambda2': [1.0, 1.2],
            'lambda3': [0.1, 0.1]
        })
        
        df_blend = stack_probabilities(df_features, df_lambda, w_raw=0.6, w_poisson=0.4)
        
        # Verificar estructura
        assert len(df_blend) == 2
        assert all(col in df_blend.columns for col in ['p_blend_L', 'p_blend_E', 'p_blend_V'])
        
        # Verificar que las probabilidades son válidas
        validar_probabilidades(df_blend, cols_prefix='p_blend')
        
        # Verificar que es una combinación lineal
        # p_blend debería estar entre p_raw y p_pois
        assert (df_blend['p_blend_L'] >= 0).all()
        assert (df_blend['p_blend_L'] <= 1).all()


class TestBayesianAdjustment:
    """Pruebas para ajuste bayesiano"""
    
    @pytest.fixture
    def sample_data_for_bayes(self):
        """Datos para pruebas bayesianas"""
        df_blend = pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'p_blend_L': [0.40, 0.35],
            'p_blend_E': [0.30, 0.30],
            'p_blend_V': [0.30, 0.35]
        })
        
        df_features = pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'delta_forma': [2.0, -1.0],
            'inj_weight': [0.1, 0.2],
            'is_final': [True, False],
            'is_derby': [False, True]
        })
        
        return df_blend, df_features
    
    def test_ajustar_bayes_estructura(self, sample_data_for_bayes):
        """Prueba estructura del ajuste bayesiano"""
        df_blend, df_features = sample_data_for_bayes
        
        df_final = ajustar_bayes(df_blend, df_features)
        
        # Verificar columnas
        assert 'p_final_L' in df_final.columns
        assert 'p_final_E' in df_final.columns
        assert 'p_final_V' in df_final.columns
        
        # Verificar que siguen siendo probabilidades válidas
        validar_probabilidades(df_final, cols_prefix='p_final')
    
    def test_ajustar_bayes_impacto_positivo(self, sample_data_for_bayes):
        """Prueba que forma positiva aumenta probabilidad local"""
        df_blend, df_features = sample_data_for_bayes
        
        # Coeficientes que favorecen al local con buena forma
        coef = {
            "k1_L": 0.15,  # Positivo para local
            "k1_E": -0.10,  # Negativo para empate
            "k1_V": -0.05,  # Negativo para visitante
            "k2_L": 0, "k2_E": 0, "k2_V": 0,  # Ignorar otros factores
            "k3_L": 0, "k3_E": 0, "k3_V": 0
        }
        
        df_final = ajustar_bayes(df_blend, df_features, coef)
        
        # Para el partido 1 con delta_forma = 2.0 (positivo)
        # p_final_L debería ser mayor que p_blend_L
        assert df_final.loc[0, 'p_final_L'] > df_blend.loc[0, 'p_blend_L']
        
        # Para el partido 2 con delta_forma = -1.0 (negativo)
        # p_final_L debería ser menor que p_blend_L
        assert df_final.loc[1, 'p_final_L'] < df_blend.loc[1, 'p_blend_L']


class TestDrawPropensity:
    """Pruebas para reglas de empate"""
    
    def test_aplicar_draw_propensity_caso_aplicable(self):
        """Prueba cuando se debe aplicar la regla"""
        df = pd.DataFrame({
            'match_id': ['2283-1', '2283-2'],
            'p_final_L': [0.34, 0.50],
            'p_final_E': [0.38, 0.25],  # Partido 1: E > max(L,V) y |L-V| < 0.08
            'p_final_V': [0.28, 0.25]
        })
        
        df_adjusted = aplicar_draw_propensity(df)
        
        # Partido 1 cumple condiciones: |0.34-0.28| = 0.06 < 0.08 y 0.38 > 0.34
        # Debería ajustarse
        assert df_adjusted.loc[0, 'p_final_E'] > df.loc[0, 'p_final_E']
        assert df_adjusted.loc[0, 'p_final_L'] < df.loc[0, 'p_final_L']
        
        # Partido 2 no cumple: |0.50-0.25| = 0.25 > 0.08
        # No debería cambiar
        assert df_adjusted.loc[1, 'p_final_E'] == df.loc[1, 'p_final_E']
        
        # Verificar que siguen siendo probabilidades válidas
        validar_probabilidades(df_adjusted, cols_prefix='p_final')
    
    def test_aplicar_draw_propensity_suma_uno(self):
        """Prueba que las probabilidades sigan sumando 1"""
        df = pd.DataFrame({
            'match_id': ['2283-1'],
            'p_final_L': [0.35],
            'p_final_E': [0.36],
            'p_final_V': [0.29]
        })
        
        df_adjusted = aplicar_draw_propensity(df)
        
        # Verificar suma = 1
        suma = df_adjusted[['p_final_L', 'p_final_E', 'p_final_V']].sum(axis=1).iloc[0]
        assert abs(suma - 1.0) < 1e-6


class TestCalibration:
    """Pruebas de calibración y métricas"""
    
    def test_log_loss_calculation(self):
        """Prueba cálculo de log-loss"""
        from sklearn.metrics import log_loss
        
        # Probabilidades predichas
        y_pred = np.array([
            [0.7, 0.2, 0.1],  # Favorito local claro
            [0.3, 0.4, 0.3],  # Partido cerrado
            [0.2, 0.3, 0.5]   # Favorito visitante
        ])
        
        # Resultados reales (one-hot)
        y_true = np.array([
            [1, 0, 0],  # Ganó local (predicción correcta)
            [0, 1, 0],  # Empate
            [0, 0, 1]   # Ganó visitante (predicción correcta)
        ])
        
        loss = log_loss(y_true, y_pred)
        
        # El loss debería ser relativamente bajo porque 2/3 predicciones fueron correctas
        assert loss < 1.0
        
    def test_brier_score(self):
        """Prueba cálculo de Brier score"""
        # Brier score = mean((pred - actual)^2)
        
        y_pred = np.array([0.8, 0.6, 0.3])
        y_true = np.array([1, 1, 0])
        
        brier = np.mean((y_pred - y_true) ** 2)
        
        # Calcular manualmente
        # (0.8-1)^2 + (0.6-1)^2 + (0.3-0)^2 = 0.04 + 0.16 + 0.09 = 0.29
        expected = 0.29 / 3
        
        assert abs(brier - expected) < 1e-6


class TestIntegrationModeling:
    """Pruebas de integración del pipeline de modelado"""
    
    def test_pipeline_modelado_completo(self):
        """Prueba el pipeline de modelado completo"""
        # Crear datos sintéticos
        n_matches = 14
        
        df_features = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, n_matches + 1)],
            'home': [f'Equipo{i}' for i in range(1, n_matches + 1)],
            'away': [f'Equipo{i+14}' for i in range(1, n_matches + 1)],
            'liga': ['Liga MX'] * n_matches,
            'p_raw_L': np.random.uniform(0.2, 0.6, n_matches),
            'p_raw_E': np.random.uniform(0.25, 0.35, n_matches),
            'p_raw_V': np.random.uniform(0.2, 0.5, n_matches),
            'goles_H': np.random.randint(0, 4, n_matches),
            'goles_A': np.random.randint(0, 4, n_matches),
            'elo_home': np.random.randint(1500, 1700, n_matches),
            'elo_away': np.random.randint(1500, 1700, n_matches),
            'factor_local': [0.45] * n_matches,
            'delta_forma': np.random.uniform(-2, 2, n_matches),
            'inj_weight': np.random.uniform(0, 0.3, n_matches),
            'is_final': np.random.choice([True, False], n_matches, p=[0.1, 0.9]),
            'is_derby': np.random.choice([True, False], n_matches, p=[0.15, 0.85])
        })
        
        # Normalizar probabilidades raw
        prob_sum = df_features[['p_raw_L', 'p_raw_E', 'p_raw_V']].sum(axis=1)
        for col in ['p_raw_L', 'p_raw_E', 'p_raw_V']:
            df_features[col] = df_features[col] / prob_sum
        
        # 1. Preparar para Poisson
        df_poisson = preparar_dataset_poisson(df_features)
        assert 'elo_diff' in df_poisson.columns
        
        # 2. Simular lambdas (en lugar de entrenar modelo real)
        df_lambda = pd.DataFrame({
            'match_id': df_features['match_id'],
            'lambda1': np.random.uniform(1.0, 2.5, n_matches),
            'lambda2': np.random.uniform(0.8, 2.2, n_matches),
            'lambda3': np.random.uniform(0.05, 0.15, n_matches)
        })
        
        # 3. Stack
        df_blend = stack_probabilities(df_features, df_lambda)
        assert validar_probabilidades(df_blend, cols_prefix='p_blend') is None
        
        # 4. Ajuste bayesiano
        df_final = ajustar_bayes(df_blend, df_features)
        assert validar_probabilidades(df_final, cols_prefix='p_final') is None
        
        # 5. Draw propensity
        df_adjusted = aplicar_draw_propensity(df_final)
        assert len(df_adjusted) == n_matches
        
        # Verificar rangos globales
        total_L = df_adjusted['p_final_L'].sum()
        total_E = df_adjusted['p_final_E'].sum()
        total_V = df_adjusted['p_final_V'].sum()
        
        print(f"Totales finales: L={total_L:.2f}, E={total_E:.2f}, V={total_V:.2f}")
        
        # Deberían estar cerca de los rangos esperados (con 14 partidos)
        assert 4.0 <= total_L <= 6.5  # Rango ampliado para datos aleatorios
        assert 2.5 <= total_E <= 5.5
        assert 3.5 <= total_V <= 6.0
        
        print("✅ Pipeline de modelado completado exitosamente")


# Configuración para pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])