import random as r

class perceptron():
    def __init__(self, numero_entradas, faixa, taxa_de_aprendizado):
        self.numero_entradas = numero_entradas
        self.faixa = faixa
        self.result = 0
        self.indici_de_desempenho = 0
        self.taxa_de_aprendizado = taxa_de_aprendizado
        self.pesos = []
        self.entradas = []
        self.gera_pesos_aleatorios()

    def gera_pesos_aleatorios(self):
        for i in range(self.numero_entradas):
            number = self.get_random_number(self.faixa)
            self.pesos.append(number)

    def get_random_number(self, faixa) -> float:
        return r.uniform(-faixa, faixa)

    def get_pesos(self) -> list:
        return self.pesos
    
    def process(self, entradas):
        self.result = 0
        self.entradas = entradas
        tam = len(self.pesos)
        for i in range(tam):
            res = entradas[i] * self.pesos[i]
            self.result = self.result + res

    def get_result(self) -> float:
        return self.result

    def compare_expected(self, expected):
        error = self.result - expected
        self.indici_de_desempenho = abs((error/2)**2)
        tam = len(self.pesos)
        for i in range(tam):
            self.pesos[i] = self.pesos[i] + (self.taxa_de_aprendizado * abs(error) * self.entradas[i])

    def get_indici_de_desempenho(self) -> float:
        return self.indici_de_desempenho

def get_instace(entradas, faixa, taxa_de_aprendizado):
    return perceptron(entradas, faixa, taxa_de_aprendizado)