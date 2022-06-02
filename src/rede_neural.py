import src.perceptron as perceptron

class rede_neural():
    def __init__(self, camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos) -> None:
        self.camadas = camadas
        self.num_entradas = num_entradas
        self.taxa_de_aprendizagem = taxa_de_aprendizagem
        self.faixa_de_pesos = faixa_de_pesos
        self.rede = []
        self.perceptron_saida = self.perceptron_factory()
        self.resultado = 0
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
        

def get_instance(camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos) -> rede_neural:
    return rede_neural(camadas, num_entradas, taxa_de_aprendizagem, faixa_de_pesos)