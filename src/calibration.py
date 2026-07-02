import numpy as np

Z_BANDA = 2.576  # ~99% assumindo resíduos aproximadamente normais


def calibrar(modelo, df_calib, features, z=Z_BANDA):
    """Calcula erros, bandas por hora e thresholds de drift numa janela.

    Retorna um dicionário autocontido: nenhuma função downstream depende
    de variáveis globais. A banda usa o desvio-padrão dos resíduos por
    hora do dia (com fallback para o desvio global quando a hora não
    apareceu na calibração).
    """
    df = df_calib.copy()
    df['yhat'] = modelo.predict(df[features])
    df['erro'] = df['y'] - df['yhat']
    df['abs_erro'] = df['erro'].abs()

    std_por_hora = df.groupby(df['ds'].dt.hour)['erro'].std()
    std_global = df['erro'].std()

    def largura_banda(ts):
        std_hora = std_por_hora.get(ts.hour, std_global)
        if np.isnan(std_hora):
            std_hora = std_global
        return z * std_hora

    df['banda'] = df['ds'].map(largura_banda)
    df['yhat_lower'] = df['yhat'] - df['banda']
    df['yhat_upper'] = df['yhat'] + df['banda']
    df['anomalia'] = (df['y'] > df['yhat_upper']) | (df['y'] < df['yhat_lower'])

    mae = df['abs_erro'].mean()
    taxa_anomalia = df['anomalia'].mean()

    return {
        'df': df,
        'erros_historicos': df['erro'].values,
        'mae_historico': mae,
        'threshold_mae': mae + 3 * df['abs_erro'].std(),
        'std_residuos': std_global,
        'threshold_residual_std': std_global * 1.5,
        'taxa_anomalia': taxa_anomalia,
        'threshold_taxa_anomalia': max(taxa_anomalia + 0.02, 0.03),
        'largura_banda': largura_banda,
        'z': z,
    }
