import unittest
import src.perceptron as perceptron

class tests_perceptron(unittest.TestCase):

    def setUp(self):
        self.faixa = 1
        self.num_entradas = 5
        self.entradas = [1, 1, 1, 1, 1]
        self.taxa_de_aprendizado = 5
        self.perceptron = perceptron.get_instace(self.num_entradas, self.faixa, self.taxa_de_aprendizado)
        self.pesos = self.perceptron.get_pesos().copy()

    def test01_generate_random_weigth(self):
        num = self.perceptron.get_random_number(self.faixa)
        self.assertLessEqual(num, self.faixa, "Numero aleatório maior que o limite superior")
        self.assertGreaterEqual(num, -self.faixa, "Numéro aleatório menor que o limite inferior")

    def test02_see_perceptron_result(self):
        sum_pesos = sum(self.pesos)
        self.perceptron.process(self.entradas)
        res = self.perceptron.get_result()
        self.assertAlmostEquals(sum_pesos, res, "Resposta diferente da esperada")

    def test03_test_perceptron_mudancas_de_pesos(self):
        expected = 2
        self.perceptron.process(self.entradas)
        self.perceptron.compare_expected(expected)
        novosPesos = self.perceptron.get_pesos()
        self.assertNotEqual(self.pesos, novosPesos, "pesos se manteram após aprendizado")

    def test04_test_indici_de_erro(self):
        expected_result = 5
        self.perceptron.process(self.entradas)
        res = self.perceptron.get_result()
        error = res - expected_result
        expected_indice = abs((error/2)**2)
        self.perceptron.compare_expected(expected_result)
        indice = self.perceptron.get_indici_de_desempenho()
        self.assertAlmostEquals(expected_indice, indice, "indice de desempenho calculado incorretamente")

if __name__ == '__main__':
    unittest.main()