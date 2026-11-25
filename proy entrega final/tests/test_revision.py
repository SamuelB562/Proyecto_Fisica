"""Pruebas de regresión de las magnitudes analíticas; no requieren Assetto LIVE."""
import unittest
import sys
import types
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
# Sustituye solamente el módulo controlador vJoy, que depende de Windows.
sys.modules.setdefault('pyvjoy', types.SimpleNamespace())

from leer_parametros_ac import limites_experimentales
from modelado_fisico import Evaluacion
from conducir_fisica_optimizado import tiempo_vuelta_plan


class RevisionExperimento(unittest.TestCase):
    def test_calibracion_coincide_con_resumen_ponderado(self):
        a, b = limites_experimentales('jw_porsche_991.2_gt2_rs_us',
                                      'Hypercar road (HR)')
        self.assertAlmostEqual(a, 6.3527021978, places=5)
        self.assertAlmostEqual(b, 9.93363125, places=5)

    def test_calibracion_no_se_asigna_a_otro_vehiculo(self):
        self.assertIsNone(limites_experimentales('otro_auto', 'Hypercar road (HR)'))

    def test_tiempo_vuelta_excluye_prefijo_y_sufijo(self):
        puntos = np.column_stack((np.arange(12., dtype=float), np.zeros(12)))
        evaluacion = Evaluacion(1.1, 11., np.full(12,10.),
                                np.zeros(12), np.zeros(12), 0.)
        self.assertAlmostEqual(tiempo_vuelta_plan(puntos,evaluacion,
                                                   np.arange(12.),3.,8.),0.5)


if __name__ == '__main__':
    unittest.main()
