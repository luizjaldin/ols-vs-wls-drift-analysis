import numpy as np
import pandas as pd

from src.drift import (
    drift_distribuicao_erros,
    drift_mae,
    drift_taxa_anomalia,
    drift_volatilidade_residuos,
)

CONFIG_PADRAO = {
    'window_size': 48,       # 24h de medições semi-horárias
    'ks_window_size': 96,    # 2 dias
    'ks_cadencia': 20,       # roda o KS a cada 20 observações
    'min_drift_signals': 3,  # de 4 sinais possíveis
}


def simular_stream(modelo, df_stream, calib, features, config=None):
    """Simula chegada sequencial e monitora 4 sinais de drift.

    - Anomalia é bilateral (fora da banda para cima ou para baixo).
    - O flag do KS persiste entre execuções do teste: o teste roda a cada
      `ks_cadencia` observações e o último veredito vale até o próximo.
    - Para na primeira observação com >= min_drift_signals sinais ativos.

    Retorna (df_monitoramento, info_drift), com info_drift = None se o
    stream terminar sem drift.
    """
    cfg = {**CONFIG_PADRAO, **(config or {})}

    erros, flags_anomalia, registros = [], [], []
    flag_ks = 0
    ks_stat, ks_p_value = np.nan, np.nan
    info_drift = None

    for pos, (_, row) in enumerate(df_stream.iterrows()):
        y_real = row['y']
        x_atual = row[features].to_numpy(dtype=float).reshape(1, -1)
        y_hat = float(modelo.predict(x_atual)[0])

        banda = calib['largura_banda'](row['ds'])
        y_lower, y_upper = y_hat - banda, y_hat + banda

        erro = float(y_real - y_hat)
        erros.append(erro)
        anomalia = bool(y_real > y_upper or y_real < y_lower)
        flags_anomalia.append(int(anomalia))

        janela_cheia = len(erros) >= cfg['window_size']
        erros_recentes = erros[-cfg['window_size']:]
        anomalias_recentes = flags_anomalia[-cfg['window_size']:]

        r_mae = drift_mae(np.abs(erros_recentes), calib['threshold_mae'])
        r_taxa = drift_taxa_anomalia(anomalias_recentes, calib['threshold_taxa_anomalia'])
        r_vol = drift_volatilidade_residuos(erros_recentes, calib['threshold_residual_std'])

        flag_mae = int(janela_cheia and r_mae['drift'])
        flag_taxa = int(janela_cheia and r_taxa['drift'])
        flag_vol = int(janela_cheia and r_vol['drift'])

        if len(erros) >= cfg['ks_window_size'] and pos % cfg['ks_cadencia'] == 0:
            r_ks = drift_distribuicao_erros(
                calib['erros_historicos'], erros[-cfg['ks_window_size']:]
            )
            ks_stat, ks_p_value = r_ks['ks_stat'], r_ks['p_value']
            flag_ks = int(r_ks['drift'])

        qtd_sinais = flag_mae + flag_taxa + flag_vol + flag_ks
        drift_na_linha = qtd_sinais >= cfg['min_drift_signals']

        registros.append({
            'posicao_stream': pos,
            'ds': row['ds'],
            'y': y_real,
            'yhat': y_hat,
            'yhat_lower': y_lower,
            'yhat_upper': y_upper,
            'erro': erro,
            'mae_movel': r_mae['mae_janela'],
            'taxa_anomalia_movel': r_taxa['taxa_anomalia'],
            'std_residuos_movel': r_vol['std_janela'],
            'ks_stat': ks_stat,
            'ks_p_value': ks_p_value,
            'flag_mae': flag_mae,
            'flag_anomalia': flag_taxa,
            'flag_volatilidade': flag_vol,
            'flag_distribuicao_erro': flag_ks,
            'qtd_sinais_drift': qtd_sinais,
            'drift_detectado': drift_na_linha,
            'anomalia': anomalia,
        })

        if drift_na_linha:
            info_drift = {'ds': row['ds'], 'posicao': pos, 'sinais': qtd_sinais}
            break

    return pd.DataFrame(registros), info_drift
