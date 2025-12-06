# IA Evolutiva - PSO/GA

1) Como instalar as dependências

```  
python -m venv .venv  
.\.venv\Scripts\Activate.ps1  
pip install -r requirements.txt  
```
O dataset já está incluído em data/wdbc.data.

2) Rodar um pequeno teste para main  

```
.\.venv\Scripts\Activate.ps1  
python main.py --data .\data\wdbc.data --out .\smoke_test --gens 3 --repeats 1
```

3) Rodar o algoritmo completo (com parâmetros do artigo) 

```
python main.py --data .\data\wdbc.data --out .\resultados --gens 300 --repeats 30
```
  
*Observação: rodar novamente sobrescreve os arquivos dentro de .\resultados.

Depois, basta modificar a pasta de origem dos resultados em plots.ipynb e gerar os gráficos de comparação.
