import numpy as np

N_LAGS = 5
JANELA_MEDIA_MOVEL = 24  # 12h — janela != span dos lags evita combinação linear

FEATURES = [
    'lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5',
    'rolling_mean_24',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
]
TARGET = 'y'


def construir_features(df):
    """Cria variáveis explicativas usando apenas valores passados.

    rolling_mean_5 foi substituída por rolling_mean_24: a média das 5
    observações anteriores é exatamente (lag_1+...+lag_5)/5, o que tornava
    a matriz de projeto singular (posto incompleto).
    """
    df = df.copy()
    for k in range(1, N_LAGS + 1):
        df[f'lag_{k}'] = df['y'].shift(k)
    df['rolling_mean_24'] = df['y'].shift(1).rolling(JANELA_MEDIA_MOVEL).mean()
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df.dropna().reset_index(drop=True)
