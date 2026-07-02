import matplotlib.pyplot as plt


def plot_serie_com_bandas(df, titulo):
    """Valor real vs previsto com bandas; anomalias (coluna 'anomalia') em vermelho."""
    normais = df[~df['anomalia']]
    anomalos = df[df['anomalia']]

    plt.figure(figsize=(15, 7))
    plt.plot(df['ds'], df['yhat'], color='#2ca02c', linewidth=2, label='Previsão')
    plt.plot(df['ds'], df['yhat_upper'], color='orange', linestyle='--', alpha=0.7,
             label='Limite superior')
    plt.plot(df['ds'], df['yhat_lower'], color='orange', linestyle='--', alpha=0.7,
             label='Limite inferior')
    plt.fill_between(df['ds'], df['yhat_lower'], df['yhat_upper'], color='orange', alpha=0.05)
    plt.scatter(normais['ds'], normais['y'], color='#1f77b4', s=25, alpha=0.7,
                label='Valor real')
    plt.scatter(anomalos['ds'], anomalos['y'], color='red', s=55, edgecolors='black',
                linewidths=1.2, zorder=5, label='Anomalia')
    plt.title(titulo, fontsize=14, fontweight='bold')
    plt.xlabel('Data/hora')
    plt.ylabel('Volume de táxis')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_metrica_movel(df, coluna, titulo, ylabel, threshold=None, drift_ts=None):
    plt.figure(figsize=(15, 5))
    plt.plot(df['ds'], df[coluna], label=ylabel)
    if threshold is not None:
        plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
    if drift_ts is not None:
        plt.axvline(drift_ts, color='black', linestyle='--', linewidth=2,
                    label='Drift detectado')
    plt.title(titulo)
    plt.xlabel('Data')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_sinais_drift(df, minimo, drift_ts=None):
    plt.figure(figsize=(15, 5))
    plt.step(df['ds'], df['qtd_sinais_drift'], where='post', label='Sinais de drift ativos')
    plt.axhline(minimo, color='red', linestyle='--', label='Mínimo para declarar drift')
    if drift_ts is not None:
        plt.axvline(drift_ts, color='black', linestyle='--', linewidth=2,
                    label='Drift detectado')
    plt.title('Evolução dos Sinais de Drift')
    plt.xlabel('Data')
    plt.ylabel('Número de sinais')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()
