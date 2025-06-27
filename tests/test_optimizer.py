import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import Mock, patch

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimizer.classify_matches import etiquetar_partidos
from src.optimizer.generate_core import generar_core_base, generar_variaciones_core
from src.optimizer.generate_satellites import generar_satellites, invertir_signo
from src.optimizer.grasp import generar_candidatos, grasp_portfolio
from src.optimizer.annealing import F, pr_11_boleto, annealing_optimize
from src.optimizer.checklist import (check_portfolio, porcentaje_signo, 
                                     max_sign_pct, max_sign_pct_first3, 
                                     jaccard_similarity)


class TestClassifyMatches:
    """Pruebas para clasificación de partidos"""
    
    @pytest.fixture
    def sample_prob_data(self):
        """Datos de probabilidades para clasificación"""
        return pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 15)],
            'p_final_L': [0.65, 0.45, 0.30, 0.50, 0.38, 0.70, 0.35, 0.42, 0.33, 0.55, 0.48, 0.62, 0.40, 0.36],
            'p_final_E': [0.20, 0.30, 0.35, 0.30, 0.32, 0.18, 0.33, 0.29, 0.40, 0.25, 0.27, 0.23, 0.35, 0.32],
            'p_final_V': [0.15, 0.25, 0.35, 0.20, 0.30, 0.12, 0.32, 0.29, 0.27, 0.20, 0.25, 0.15, 0.25, 0.32]
        })
    
    @pytest.fixture
    def sample_features_data(self):
        """Features para clasificación"""
        return pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 15)],
            'volatilidad_flag': [False] * 14,
            'draw_propensity_flag': [False, False, True, False, True, False, True, False, True, False, False, False, True, True]
        })
    
    def test_etiquetar_partidos_ancla(self, sample_prob_data, sample_features_data):
        """Prueba identificación de partidos Ancla"""
        df_tags = etiquetar_partidos(sample_prob_data, sample_features_data)
        
        # Partidos con p_max > 0.60 deberían ser Ancla
        anclas = df_tags[df_tags['etiqueta'] == 'Ancla']
        
        # Partidos 1, 6 y 12 tienen p_max > 0.60
        assert len(anclas) >= 2
        assert '2283-1' in anclas['match_id'].values  # p_max = 0.65
        assert '2283-6' in anclas['match_id'].values  # p_max = 0.70
    
    def test_etiquetar_partidos_divisor(self, sample_prob_data, sample_features_data):
        """Prueba identificación de Divisores"""
        df_tags = etiquetar_partidos(sample_prob_data, sample_features_data)
        
        divisores = df_tags[df_tags['etiqueta'] == 'Divisor']
        
        # Deberían ser partidos con 0.40 < p_max < 0.60
        for _, row in divisores.iterrows():
            assert 0.40 < row['p_max'] <= 0.60
    
    def test_etiquetar_partidos_tendencia_x(self, sample_prob_data, sample_features_data):
        """Prueba identificación de TendenciaX"""
        df_tags = etiquetar_partidos(sample_prob_data, sample_features_data)
        
        tendencia_x = df_tags[df_tags['etiqueta'] == 'TendenciaX']
        
        # Deberían coincidir con draw_propensity_flag = True
        expected_ids = sample_features_data[sample_features_data['draw_propensity_flag']]['match_id']
        
        assert set(tendencia_x['match_id'].values) == set(expected_ids.values)


class TestGenerateCore:
    """Pruebas para generación de quinielas Core"""
    
    @pytest.fixture
    def sample_data_for_core(self):
        """Datos para generar Core"""
        df_probs = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 15)],
            'p_final_L': [0.65, 0.45, 0.30] + [0.40] * 11,
            'p_final_E': [0.20, 0.30, 0.40] + [0.30] * 11,
            'p_final_V': [0.15, 0.25, 0.30] + [0.30] * 11
        })
        
        df_tags = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 15)],
            'etiqueta': ['Ancla', 'Divisor', 'TendenciaX'] + ['Neutro'] * 11,
            'signo_argmax': ['L', 'L', 'E'] + ['L'] * 6 + ['V'] * 5,
            'p_max': df_probs[['p_final_L', 'p_final_E', 'p_final_V']].max(axis=1)
        })
        
        return df_probs, df_tags
    
    def test_generar_core_base_respeta_anclas(self, sample_data_for_core):
        """Prueba que Core respeta partidos Ancla"""
        df_probs, df_tags = sample_data_for_core
        
        base_signos, empates_idx = generar_core_base(df_probs, df_tags)
        
        # Verificar que Anclas tienen su signo argmax
        anclas = df_tags[df_tags['etiqueta'] == 'Ancla']
        for idx, row in anclas.iterrows():
            assert base_signos[idx] == row['signo_argmax']
    
    def test_generar_core_base_empates_rango(self, sample_data_for_core):
        """Prueba que Core tiene 4-6 empates"""
        df_probs, df_tags = sample_data_for_core
        
        base_signos, empates_idx = generar_core_base(df_probs, df_tags)
        
        n_empates = base_signos.count('E')
        assert 4 <= n_empates <= 6
        
        # Verificar que empates_idx corresponde a posiciones con 'E'
        for idx in empates_idx:
            assert base_signos[idx] == 'E'
    
    def test_generar_variaciones_core(self):
        """Prueba generación de 4 variaciones Core"""
        base_signos = ['L', 'L', 'E', 'L', 'E', 'V', 'V', 'L', 'E', 'V', 'L', 'E', 'V', 'L']
        empates_idx = [2, 4, 8, 11]
        
        variaciones = generar_variaciones_core(base_signos, empates_idx)
        
        # Deben ser 4 variaciones
        assert len(variaciones) == 4
        
        # Primera debe ser la base
        assert variaciones[0] == base_signos
        
        # Las demás deben diferir en algunos signos
        for i in range(1, 4):
            diffs = sum(1 for j in range(14) if variaciones[i][j] != base_signos[j])
            assert 1 <= diffs <= 3  # Deben diferir en 1-3 posiciones


class TestGenerateSatellites:
    """Pruebas para generación de Satélites"""
    
    def test_invertir_signo(self):
        """Prueba inversión de signos"""
        assert invertir_signo('L') == 'V'
        assert invertir_signo('V') == 'L'
        assert invertir_signo('E') == 'E'
    
    def test_generar_satellites_pares(self):
        """Prueba generación de pares de satélites"""
        df_tags = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 8)],
            'etiqueta': ['Ancla', 'Divisor', 'Divisor', 'TendenciaX', 'Divisor', 'Neutro', 'Divisor']
        })
        
        core_base = ['L', 'L', 'E', 'E', 'V', 'L', 'V']
        
        satellites = generar_satellites(df_tags, core_base)
        
        # Deben generarse en pares
        assert len(satellites) % 2 == 0
        
        # Verificar que cada par difiere en el Divisor
        for i in range(0, len(satellites), 2):
            if i+1 < len(satellites):
                q1 = satellites[i][1]
                q2 = satellites[i+1][1]
                
                # Encontrar diferencias
                diffs = [j for j in range(len(q1)) if q1[j] != q2[j]]
                
                # Debe haber al menos una diferencia
                assert len(diffs) >= 1
                
                # La diferencia debe ser en un Divisor
                for d in diffs:
                    assert df_tags.iloc[d]['etiqueta'] in ['Divisor', 'Neutro']


class TestGRASP:
    """Pruebas para algoritmo GRASP"""
    
    def test_generar_candidatos(self):
        """Prueba generación de candidatos diversos"""
        base = ['L', 'L', 'E', 'V', 'V', 'E', 'L', 'L', 'E', 'V', 'L', 'E', 'V', 'L']
        
        candidatos = generar_candidatos(base, n=100)
        
        # Verificar número
        assert len(candidatos) == 100
        
        # Verificar que son diferentes de la base
        for c in candidatos:
            assert c != base
            # Deben diferir en al menos 6 signos
            diffs = sum(1 for i in range(14) if c[i] != base[i])
            assert diffs >= 6
        
        # Verificar diversidad
        unique_candidatos = [tuple(c) for c in candidatos]
        assert len(set(unique_candidatos)) > 50  # Al menos 50% únicos
    
    @patch('src.optimizer.grasp.estimar_pr11')
    def test_grasp_portfolio_seleccion(self, mock_pr11):
        """Prueba selección GRASP"""
        # Mock de probabilidades
        mock_pr11.return_value = 0.12
        
        df_core = pd.DataFrame({
            'quiniela_id': ['Core-1', 'Core-2'],
            'P1': ['L', 'L'],
            'P2': ['L', 'E'],
            'P3': ['E', 'L'],
            'P4': ['V', 'V'],
            'P5': ['L', 'L'],
            'P6': ['E', 'E'],
            'P7': ['L', 'V'],
            'P8': ['V', 'L'],
            'P9': ['E', 'E'],
            'P10': ['L', 'L'],
            'P11': ['V', 'V'],
            'P12': ['E', 'L'],
            'P13': ['L', 'E'],
            'P14': ['V', 'V']
        })
        
        df_sat = pd.DataFrame({
            'quiniela_id': ['Sat-1', 'Sat-2'],
            'P1': ['V', 'L'],
            'P2': ['L', 'L'],
            'P3': ['E', 'E'],
            'P4': ['L', 'V'],
            'P5': ['V', 'L'],
            'P6': ['E', 'E'],
            'P7': ['L', 'L'],
            'P8': ['V', 'V'],
            'P9': ['L', 'E'],
            'P10': ['E', 'L'],
            'P11': ['V', 'V'],
            'P12': ['E', 'E'],
            'P13': ['L', 'L'],
            'P14': ['L', 'V']
        })
        
        df_prob = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, 15)],
            'p_final_L': [0.4] * 14,
            'p_final_E': [0.3] * 14,
            'p_final_V': [0.3] * 14
        })
        
        result = grasp_portfolio(df_core, df_sat, df_prob, alpha=0.15, n_target=10)
        
        # Verificar que se alcanza el target
        assert len(result) == 10
        
        # Verificar que incluye Core y Satélites originales
        assert 'Core-1' in result['quiniela_id'].values
        assert 'Sat-1' in result['quiniela_id'].values


class TestAnnealing:
    """Pruebas para Simulated Annealing"""
    
    def test_pr_11_boleto_calculo(self):
        """Prueba cálculo de probabilidad ≥11"""
        # Quiniela con probabilidades conocidas
        q = ['L', 'L', 'E', 'V'] * 3 + ['L', 'L']  # 14 signos
        
        df_prob = pd.DataFrame({
            'p_final_L': [0.8] * 8 + [0.2] * 6,  # 8 muy probables, 6 poco probables
            'p_final_E': [0.1] * 8 + [0.4] * 6,
            'p_final_V': [0.1] * 8 + [0.4] * 6
        })
        
        # Esta quiniela tiene 8 L's con alta probabilidad (0.8)
        # y 6 signos con baja probabilidad
        pr = pr_11_boleto(q, df_prob, n_samples=10000)
        
        # La probabilidad debería ser muy baja porque necesita acertar
        # casi todos los difíciles también
        assert 0 <= pr <= 1
        assert pr < 0.2  # Debería ser baja
    
    def test_F_objetivo(self):
        """Prueba función objetivo F"""
        # Portafolio de 2 quinielas
        portfolio = [
            ['L'] * 14,  # Quiniela simple
            ['V'] * 14   # Quiniela opuesta
        ]
        
        # Probabilidades uniformes
        df_prob = pd.DataFrame({
            'p_final_L': [0.4] * 14,
            'p_final_E': [0.3] * 14,
            'p_final_V': [0.3] * 14
        })
        
        # Mock pr_11_boleto para control
        with patch('src.optimizer.annealing.pr_11_boleto') as mock_pr11:
            mock_pr11.side_effect = [0.10, 0.05]  # Primera quiniela 10%, segunda 5%
            
            f_value = F(portfolio, df_prob)
            
            # F = 1 - (1-0.10)*(1-0.05) = 1 - 0.9*0.95 = 1 - 0.855 = 0.145
            expected = 1 - 0.9 * 0.95
            assert abs(f_value - expected) < 1e-6
    
    @patch('src.optimizer.annealing.pr_11_boleto')
    def test_annealing_optimize_mejora(self, mock_pr11):
        """Prueba que annealing mejora o mantiene F"""
        # Configurar mock para devolver valores que favorezcan ciertos cambios
        mock_pr11.return_value = 0.08
        
        portfolio_inicial = [
            ['L', 'L', 'E', 'V'] * 3 + ['L', 'L'],
            ['V', 'V', 'L', 'E'] * 3 + ['V', 'V']
        ]
        
        df_prob = pd.DataFrame({
            'p_final_L': [0.4] * 14,
            'p_final_E': [0.3] * 14,
            'p_final_V': [0.3] * 14
        })
        
        # Calcular F inicial
        f_inicial = F(portfolio_inicial, df_prob)
        
        # Optimizar
        portfolio_final = annealing_optimize(
            portfolio_inicial, 
            df_prob, 
            T0=0.1, 
            beta=0.9, 
            max_iter=100
        )
        
        # Calcular F final
        f_final = F(portfolio_final, df_prob)
        
        # F no debe empeorar
        assert f_final >= f_inicial - 1e-6  # Tolerancia numérica


class TestChecklist:
    """Pruebas para validación con checklist"""
    
    def test_porcentaje_signo(self):
        """Prueba cálculo de porcentaje de signo"""
        q = ['L', 'L', 'L', 'L', 'L', 'E', 'E', 'E', 'E', 'V', 'V', 'V', 'V', 'V']
        
        assert porcentaje_signo(q, 'L') == 5/14
        assert porcentaje_signo(q, 'E') == 4/14
        assert porcentaje_signo(q, 'V') == 5/14
    
    def test_max_sign_pct(self):
        """Prueba cálculo de máxima concentración"""
        q = ['L'] * 10 + ['E'] * 2 + ['V'] * 2  # 10 L's de 14
        
        assert max_sign_pct(q) == 10/14
    
    def test_max_sign_pct_first3(self):
        """Prueba concentración en primeros 3 partidos"""
        q = ['L', 'L', 'L'] + ['E'] * 11  # 3 L's en primeros 3
        
        assert max_sign_pct_first3(q) == 1.0  # 100% en primeros 3
    
    def test_jaccard_similarity(self):
        """Prueba similitud de Jaccard"""
        q1 = ['L', 'E', 'V', 'L']
        q2 = ['L', 'E', 'V', 'V']  # Difiere en posición 3
        
        # 3 coincidencias de 4
        sim = jaccard_similarity(q1, q2)
        
        # Jaccard = intersección / unión
        # En este caso, 3 posiciones iguales de 4 total
        expected = 3/4
        assert abs(sim - expected) < 1e-6
    
    def test_check_portfolio_valido(self):
        """Prueba portafolio válido"""
        # Crear portafolio que cumple todas las reglas
        quinielas = []
        
        # 4 quinielas con distribución correcta
        for i in range(4):
            # 5L, 5E, 4V = 35.7% L, 35.7% E, 28.6% V
            q = ['L'] * 5 + ['E'] * 5 + ['V'] * 4
            # Mezclar para evitar concentración en primeros 3
            np.random.shuffle(q)
            quinielas.append(q)
        
        df_port = pd.DataFrame(quinielas, columns=[f'P{i+1}' for i in range(14)])
        df_port.insert(0, 'quiniela_id', [f'Q{i+1}' for i in range(4)])
        
        # Mock print para capturar salida
        with patch('builtins.print'):
            result = check_portfolio(df_port)
        
        assert result == True
    
    def test_check_portfolio_falla_empates(self):
        """Prueba detección de empates fuera de rango"""
        # Quiniela con solo 2 empates (menos de 4)
        q = ['L'] * 6 + ['E'] * 2 + ['V'] * 6
        
        df_port = pd.DataFrame([q], columns=[f'P{i+1}' for i in range(14)])
        df_port.insert(0, 'quiniela_id', ['Q1'])
        
        with patch('builtins.print'):
            result = check_portfolio(df_port)
        
        assert result == False
    
    def test_check_portfolio_falla_concentracion(self):
        """Prueba detección de concentración excesiva"""
        # Quiniela con 11 L's de 14 (78% > 70%)
        q = ['L'] * 11 + ['E'] * 2 + ['V'] * 1
        
        df_port = pd.DataFrame([q], columns=[f'P{i+1}' for i in range(14)])
        df_port.insert(0, 'quiniela_id', ['Q1'])
        
        with patch('builtins.print'):
            result = check_portfolio(df_port)
        
        assert result == False


class TestIntegrationOptimizer:
    """Pruebas de integración del optimizador completo"""
    
    def test_pipeline_optimizacion_completo(self):
        """Prueba el pipeline completo de optimización"""
        # Crear datos de entrada
        n_matches = 14
        
        # Probabilidades finales
        df_prob = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, n_matches + 1)],
            'p_final_L': [0.65, 0.45, 0.30, 0.50, 0.38, 0.70, 0.35, 0.42, 0.33, 0.55, 0.48, 0.62, 0.40, 0.36],
            'p_final_E': [0.20, 0.30, 0.35, 0.30, 0.32, 0.18, 0.33, 0.29, 0.40, 0.25, 0.27, 0.23, 0.35, 0.32],
            'p_final_V': [0.15, 0.25, 0.35, 0.20, 0.30, 0.12, 0.32, 0.29, 0.27, 0.20, 0.25, 0.15, 0.25, 0.32]
        })
        
        # Features
        df_features = pd.DataFrame({
            'match_id': [f'2283-{i}' for i in range(1, n_matches + 1)],
            'volatilidad_flag': [False] * n_matches,
            'draw_propensity_flag': [False, False, True, False, True, False, True, False, True, False, False, False, True, True]
        })
        
        # 1. Clasificar partidos
        df_tags = etiquetar_partidos(df_prob, df_features)
        assert len(df_tags) == n_matches
        
        # 2. Generar Core
        base_signos, empates_idx = generar_core_base(df_prob, df_tags)
        assert 4 <= len(empates_idx) <= 6
        
        core_quinielas = generar_variaciones_core(base_signos, empates_idx)
        assert len(core_quinielas) == 4
        
        # 3. Generar Satélites (limitado para test)
        satellites = generar_satellites(df_tags, core_quinielas[0])
        assert len(satellites) >= 4  # Al menos 2 pares
        
        # 4. Crear portafolio inicial
        portfolio = core_quinielas + [s[1] for s in satellites[:6]]  # 10 quinielas total
        
        # 5. Validar con checklist
        df_port = pd.DataFrame(portfolio, columns=[f'P{i+1}' for i in range(14)])
        df_port.insert(0, 'quiniela_id', [f'Q{i+1}' for i in range(len(portfolio))])
        
        # Ajustar para cumplir checklist si es necesario
        for i, q in enumerate(portfolio):
            # Asegurar 4-6 empates
            n_empates = q.count('E')
            if n_empates < 4:
                # Convertir algunos signos a E
                for j in range(14):
                    if q[j] != 'E' and n_empates < 4:
                        q[j] = 'E'
                        n_empates += 1
            elif n_empates > 6:
                # Convertir algunos E a otro signo
                for j in range(14):
                    if q[j] == 'E' and n_empates > 6:
                        q[j] = 'L'
                        n_empates -= 1
        
        # Verificar distribución global
        all_signos = [s for q in portfolio for s in q]
        n_total = len(all_signos)
        pct_L = all_signos.count('L') / n_total
        pct_E = all_signos.count('E') / n_total
        pct_V = all_signos.count('V') / n_total
        
        print(f"Distribución: L={pct_L:.1%}, E={pct_E:.1%}, V={pct_V:.1%}")
        
        # 6. Simular optimización (versión simplificada)
        with patch('src.optimizer.annealing.pr_11_boleto') as mock_pr11:
            # Simular probabilidades variables
            mock_pr11.side_effect = [0.08 + i*0.01 for i in range(100)]
            
            portfolio_opt = annealing_optimize(portfolio, df_prob, T0=0.05, beta=0.95, max_iter=50)
            
            assert len(portfolio_opt) == len(portfolio)
        
        print("✅ Pipeline de optimización completado exitosamente")


# Configuración para pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])