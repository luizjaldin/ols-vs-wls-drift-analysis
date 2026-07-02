# OLS vs WLS — Detecção e recuperação de drift em séries temporais

Trabalho da disciplina MAI5002 (Fundamentos de Matemática Aplicada).
Treina um modelo de mínimos quadrados ordinários (OLS) numa janela limitada
da série NYC Taxi (NAB), monitora sinais de drift em um stream simulado
(MAE móvel, taxa de anomalias, volatilidade dos resíduos, teste KS) e, ao
detectar drift, retreina com mínimos quadrados ponderados (WLS), comparando
pesos por erro e por recência contra o controle de um OLS retreinado.

## Estrutura

- `ols_vs_wls_drift_analysis_v2.ipynb` — notebook principal (narrativa + resultados)
- `src/` — modelos, features, calibração, sinais de drift, simulação, plots (importados pelo notebook)
- `ols_vs_wls_drift_analysis.ipynb` — versão original (referência histórica)

## Como executar

1. O dataset NAB já está incluído em `nab/` (fonte:
   https://www.kaggle.com/datasets/boltzmannbrain/nab/data).
2. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
3. Abrir `ols_vs_wls_drift_analysis_v2.ipynb` no Jupyter (a partir da raiz do
   projeto, para os imports de `src/` resolverem) e executar; ou
   `.venv/bin/jupyter nbconvert --to notebook --execute --inplace ols_vs_wls_drift_analysis_v2.ipynb`

## Trabalhos futuros

- IRLS (reponderação iterativa) no retreino
- Bandas adaptativas recalculadas com resíduos recentes do stream
- Resolução das equações normais via QR/Cholesky com análise de estabilidade
