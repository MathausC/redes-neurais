# Redes Neurais – Previsão das Ações do Banco do Brasil (BBAS3)

Projeto acadêmico em Python que implementa **uma rede neural do zero** (sem TensorFlow, PyTorch ou scikit-learn) e a treina com o histórico de cotações da ação **BBAS3 (Banco do Brasil)** para tentar prever o valor de fechamento **7 pregões à frente**.

O objetivo do projeto é didático: entender por dentro como funcionam perceptrons, camadas, pesos e ajuste por erro, implementando cada peça manualmente e cobrindo-as com testes automatizados.

> **Status:** projeto de estudo de 2022, não mantido ativamente. Veja a seção [Estado atual e limitações](#estado-atual-e-limitações) para detalhes do que funciona e do que precisa de ajuste.

## Estrutura do repositório

```
.
├── src/
│   ├── perceptron.py     # Neurônio individual: pesos, soma ponderada, erro e ajuste
│   ├── rede_neural.py    # Rede com N camadas de perceptrons + perceptron de saída
│   └── teacher.py        # Lê o CSV, treina a rede, e plota resultado vs. esperado
├── tests/
│   ├── test_perceptron.py
│   ├── test_rede_neural.py
│   └── test_teacher.py   # placeholder (ainda sem testes)
├── files/
│   └── Histórico de Cotações-20220601.csv   # cotações diárias da BBAS3 (2012–2022)
└── LICENSE.md            # GPL-2.0
```

## Como funciona

### `Perceptron`
- Recebe `numero_entradas`, `faixa` e `taxa_de_aprendizado`.
- Inicializa os pesos aleatoriamente no intervalo `[-faixa, faixa]`.
- `process(entradas)` calcula a soma ponderada das entradas (saída linear, sem função de ativação).
- `compare_expected(esperado)` calcula o erro, atualiza o **índice de desempenho** (`(erro/2)²`) e ajusta os pesos proporcionalmente à taxa de aprendizado.

### `RedeNeural`
- Monta `camadas` camadas, cada uma com `num_entradas` perceptrons, mais um perceptron de saída.
- `process(entrada)` propaga os valores camada a camada até o perceptron de saída.
- `compare_expected(esperado)` retropropaga o ajuste da saída até a primeira camada, atualizando os pesos de cada perceptron.

### `Teacher`
- Lê o CSV com **pandas**.
- Para cada dia, usa quatro valores da linha como entrada e, como valor esperado, o fechamento **7 linhas (pregões) adiante**.
- Treina até atingir o limite de épocas ou até o erro médio piorar mais vezes que o permitido (`pioras`, uma forma simples de parada antecipada).
- Usa **matplotlib** para plotar o resultado obtido contra o esperado.

Configuração padrão usada em `teacher.py`:

```python
teacher(camadas=2, num_entradas=4, taxa=0.00001, faixa=0.5, epocas=100, pioras=30)
```

## Como executar

Requisitos: Python 3 e as bibliotecas `pandas`, `matplotlib` e `pytest` (opcional, para testes).

```bash
git clone https://github.com/MathausC/redes-neurais.git
cd redes-neurais
pip install pandas matplotlib pytest
```

**Treinar a rede e ver o gráfico** (a partir da raiz do projeto):

```bash
PYTHONPATH=src python src/teacher.py
```

**Rodar os testes** (escritos com `unittest`, compatíveis com `pytest`):

```bash
PYTHONPATH=src python -m pytest -v
```

O `PYTHONPATH=src` é necessário porque os módulos em `src/` importam uns aos outros pelo nome (`import perceptron`).

## Testes

Os testes cobrem o comportamento básico dos componentes:

- **Perceptron:** pesos gerados dentro da faixa, resultado como soma ponderada, mudança dos pesos após aprendizado e cálculo correto do índice de desempenho.
- **Rede neural:** criação de perceptrons, construção com o número correto de camadas e neurônios, saída para entrada nula e convergência após repetidas correções.

## Estado atual e limitações

O projeto foi escrito em 2022 e não foi atualizado desde então. Ao executá-lo hoje (Python 3.12, pandas 3.x), verifiquei que:

- **Testes:** 3 passam e 5 falham. Os motivos são de compatibilidade e organização, não da lógica em si: os aliases `assertEquals`/`assertAlmostEquals` foram removidos do `unittest` no Python 3.12, e há duas formas de importar o mesmo módulo (`src.perceptron` e `perceptron`), o que faz o `isinstance` falhar.
- **`teacher.py`:** o acesso posicional a valores com `iloc[linha][coluna]` em uma `Series` não funciona nas versões recentes do pandas.
- **Dados:** o CSV exportado usa vírgula como separador decimal, o que desalinha as colunas. Falta uma etapa de limpeza e normalização antes do treino.
- **Modelo:** a rede usa saída linear (sem função de ativação) e ajuste de pesos simplificado, então não é uma retropropagação completa com gradiente. Não há divisão treino/teste nem métrica de avaliação além do índice de desempenho, portanto **os resultados não devem ser usados para decisões de investimento**.

## Próximos passos possíveis

- [ ] Trocar os aliases removidos (`assertEquals` → `assertEqual`, etc.) e padronizar os imports.
- [ ] Ajustar o acesso ao DataFrame para pandas recente (`iloc[linha, coluna]`).
- [ ] Tratar o CSV (decimais, colunas vazias) e normalizar os dados.
- [ ] Adicionar função de ativação e retropropagação com gradiente.
- [ ] Separar treino e teste e medir erro (RMSE/MAE).
- [ ] Escrever os testes do `Teacher` e configurar CI com GitHub Actions.

## Licença

Distribuído sob a licença **GPL-2.0**. Veja o arquivo [LICENSE.md](LICENSE.md).
