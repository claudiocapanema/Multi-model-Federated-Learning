import numpy as np
from scipy.stats import ks_2samp


def detect_drift_ks(losses, window=5, alpha=0.05):
    """
    Detecta drift usando teste de Kolmogorov-Smirnov.
    Apenas aumentos são considerados drift.

    Args:
        losses (list[float]): histórico de valores de loss.
        window (int): tamanho da janela de comparação.
        alpha (float): nível de significância.

    Returns:
        bool: True se houve drift, False caso contrário.
    """
    if len(losses) < window + 1:
        return False  # histórico insuficiente

    history = np.array(losses[-(window + 1):-1])
    last = np.array([losses[-1]] * len(history))  # simula distribuição do último valor

    stat, p_value = ks_2samp(history, last)

    # KS detecta diferença significativa (p < alpha)
    # mas só consideramos se for aumento
    return (p_value < alpha) and (losses[-1] > history.mean())


# Exemplo
# losses = [0.239873, 0.222901, 0.222539, 0.216493, 0.217351, 0.215488, 0.205237, 0.201322, 0.194011, 0.192781, 0.197502, 0.193093, 0.18085, 0.18513, 0.194035, 0.182747, 0.188482, 0.183523, 0.181978, 0.300681]
losses =  [1.6160147823824371, 1.0478063383933056, 1.2937330977965462, 1.3190818755171196, 1.0170338201436517, 0.9200679877070503, 0.8791580479357782, 1.1626055368811117, 1.0432334102880543, 1.197636996907151, 1.1002334860672016, 1.1624946456295506, 0.7341080386357643, 1.0983246312578627, 0.789227055385345, 0.6739423651036619, 0.7193610471574852, 1.3005813792445513, 0.9708273184465871, 1.3179863194624584]
# print(len(losses))
# exit()
print(detect_drift_ks(losses))  # True (mudança significativa e positiva)
