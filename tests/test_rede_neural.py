import unittest

from src.perceptron import perceptron
import src.rede_neural as rede_neural

class test_rede_neural(unittest.TestCase):

    def setUp(self):
        self.camadas = 2
        self.num_estradas = 5
        self.taxa_de_aprendizagem = 0.1
        self.faixa_de_pesos = 1
        self.rede_neural = rede_neural.get_instance(
            self.camadas,
            self.num_estradas,
            self.taxa_de_aprendizagem,
            self.faixa_de_pesos
        )

    def test01_perceptron_factory(self):
        per = self.rede_neural.perceptron_factory()
        self.assertIsInstance(per, perceptron)

    def test02_build_network(self):
        rede = self.rede_neural.rede.copy()
        self.assertEquals(len(rede), self.camadas)
        for c in range(self.camadas):
            self.assertEquals(len(rede[c]), self.num_estradas)
            for i in range(self.num_estradas):
                self.assertIsInstance(rede[c][i], perceptron)

    def test03_get_result_rede_neural(self):
        expected = 0
        entradas = [0, 0, 0, 0, 0]
        self.rede_neural.process(entradas)
        res = self.rede_neural.get_result()
        self.assertAlmostEqual(expected, res)

    def test04_verify_aprentency(self):
        expected = 3
        entrada = [1, 1, 1, 1, 1]
        self.rede_neural.process(entrada)
        res1 = self.rede_neural.get_result()
        for i in range(100):
            self.rede_neural.compare_expected(expected)
            self.rede_neural.process(entrada)
            res2 = self.rede_neural.get_result()
        self.assertAlmostEquals(res1, res2)



if __name__ == '__main__':
    unittest.main()