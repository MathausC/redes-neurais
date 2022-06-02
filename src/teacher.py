import rede_neural
import pandas as pd
import matplotlib.pyplot as plt

class teacher():
    def __init__(self, camadas, num_entradas, taxa, faixa, epocas, pioras) -> None:
        self.epocas = epocas
        self.pioras = pioras
        self.res = []
        self.expecteds = []
        self.dias = []
        self.rede = rede_neural.get_instance(camadas, num_entradas, taxa, faixa)
    
    def teach(self, file):
        arquivo = pd.read_csv(file)
        linhas = len(arquivo)
        e_anterior = 0
        e_medio = 5
        cont = 0
        p = 0
        while cont < linhas:
            if cont > 0:
                entrada = [float(arquivo.iloc[cont][1]), float(arquivo.iloc[cont][2]), float(arquivo.iloc[cont][3]), float(arquivo.iloc[cont][4])]
                print(entrada)
                self.dias.append(arquivo.iloc[cont][0])
                self.rede.process(entrada)
                self.res.append(self.rede.get_result())
                self.expecteds.append(arquivo.iloc[cont+7][5])
                self.rede.compare_expected(float(arquivo.iloc[cont+7][5]))
                e_anterior = e_medio
                e_medio = (self.rede.get_indici_de_desempenho())/cont
                if cont > self.epocas:
                    break
                if abs(e_anterior) < abs(e_medio):
                    p = p + 1
                if p > self.pioras:
                    break
            cont = cont + 1
    
    def plot(self):
        plt.plot(self.dias, self.res, label='Resultados Obtidos', color='blue')
        plt.plot(self.dias, self.expecteds, label='resultados Esperados', color='red')

        plt.ylabel('Valores de fechamento')
        plt.ylabel('dias do ano')
        plt.title('Aprendizagem da Rede Neural')

        plt.legend()
        plt.show()
               

t = teacher(2, 4, 0.00001, 0.5, 100, 30)

def run():
    t.teach('files/Histórico de Cotações-20220601.csv')
    t.plot()

if __name__ == '__main__':
    run()