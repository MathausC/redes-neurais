import perceptron

class rede_neural():
    def __init__(self, camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos) -> None:
        self.camadas = camadas
        self.num_entradas = num_entradas
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.faixa_de_pesos = faixa_de_pesos
        self.rede = []
        self.perceptron_saida = self.perceptron_factory()
        self.resultado = 0
        self.indici_de_desempenho = 0
        self.build()

    def build(self) -> None:
        for c in range(self.camadas):
            camada = []
            for e in range(self.num_entradas):
                p = self.perceptron_factory()
                camada.append(p)
            self.rede.append(camada.copy())
    
    def perceptron_factory(self) -> perceptron:
        p = perceptron.get_instace(self.num_entradas,
            self.faixa_de_pesos,
            self.taxa_de_aprendizagem)
        return p

    def process(self, entrada) -> None:
        self.resultado = 0
        camada = len(self.rede)
        for c in range(camada):
            perc = len(self.rede[c])
            res =[]
            for i in range(perc):
                self.rede[c][i].process(entrada)
                res.append(self.rede[c][i].get_result())
            entrada = res.copy()
        self.perceptron_saida.process(entrada)
        self.resultado = self.perceptron_saida.get_result()
    
    def get_result(self):
        return self.resultado

    def compare_expected(self, expected):
        self.perceptron_saida.compare_expected(expected)
        pesos = self.perceptron_saida.get_pesos()
        camada = len(self.rede)
        while camada-1 >= 0:
            pesos_anteriores = [0] * self.num_entradas
            perc = len(self.rede[camada-1])
            while perc-1 >= 0:
                self.rede[camada-1][perc-1].compare_expected(pesos[perc-1])
                desempenho = self.rede[camada-1][perc-1].get_indici_de_desempenho()
                print(desempenho)
                novos_pesos = self.rede[camada-1][perc-1].get_pesos()
                pesos_anteriores = self.sum_arrays(pesos_anteriores, novos_pesos)
                perc = perc - 1
            pesos = pesos_anteriores.copy()
            camada = camada - 1
    
    def sum_arrays(self, array1, array2) -> list:
        tam = len(array2)
        while tam-1 >= 0:
            array1[tam-1] = array1[tam-1] + array2[tam-1]
            tam = tam - 1
        return array1
    
    def get_indici_de_desempenho(self) -> float:
        return self.perceptron_saida.get_indici_de_desempenho()

def get_instance(camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos) -> rede_neural:
    return rede_neural(camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos)