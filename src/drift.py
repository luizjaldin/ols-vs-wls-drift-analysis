import numpy as np
from scipy.stats import ks_2samp


def drift_distribuicao_erros(erros_historicos, erros_janela, alpha=0.05):
    """Teste KS: a distribuição dos erros recentes mudou vs a calibração?"""
    stat, p_value = ks_2samp(erros_historicos, erros_janela)
    return {'ks_stat': stat, 'p_value': p_value, 'drift': bool(p_value < alpha)}


def drift_mae(erros_janela_abs, threshold_mae):
    mae_janela = np.mean(erros_janela_abs)
    return {'mae_janela': mae_janela, 'drift': bool(mae_janela > threshold_mae)}


def drift_taxa_anomalia(flags_anomalia, threshold_taxa_anomalia):
    taxa = np.mean(flags_anomalia)
    return {'taxa_anomalia': taxa, 'drift': bool(taxa > threshold_taxa_anomalia)}


def drift_volatilidade_residuos(erros_janela, threshold_residual_std):
    std_janela = np.std(erros_janela)
    return {'std_janela': std_janela, 'drift': bool(std_janela > threshold_residual_std)}
