from typing_extensions import Self
import unittest

from numpy import number
import src.perceptron as perceptron

class tests_perceptron(unittest.TestCase):

    def setUp(self):
        self.faixa = 1
        self.entradas = 5
        self.taxa_de_aprendizado = 5
        self.perceptron = perceptron.get_instace(self.entradas, self.faixa, self.taxa_de_aprendizado)

    def test01_generate_random_weigth(self):
        number = self.perceptron.get_random_number(self.faixa)
        self.assertLessEqual(number, self.faixa)
        self.assertGreaterEqual(number, -self.faixa)

    def test02_see_perceptron_result(self):
        pesos = self.perceptron.get_pesos()
        sum_pesos = sum(pesos)
        entradas = [1, 1, 1, 1, 1]
        self.perceptron.process(entradas)
        res = self.perceptron.get_result()
        self.assertAlmostEquals(sum_pesos, res)

    def test03_test_perceptron_output(self):
        pesos = self.perceptron.get_pesos().copy()
        entradas = [1, 1, 1, 1, 1]
        expected = 2
        self.perceptron.process(entradas)
        self.perceptron.compare_expected(expected)
        novosPesos = self.perceptron.get_pesos()
        self.assertNotEqual(pesos, novosPesos, "pesos se manteram após aprendizado")

if __name__ == '__main__':
    unittest.main()