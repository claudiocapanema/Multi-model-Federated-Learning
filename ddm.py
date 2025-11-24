import math

class DDM:
    """
    Drift Detection Method (DDM)
    Reference:
    Gama, J., Medas, P., Castillo, G., & Rodrigues, P. (2004).
    Learning with Drift Detection. Brazilian Symposium on Artificial Intelligence.
    """

    def __init__(self, warning_level=2.0, drift_level=3.0):
        self.n = 0  # number of samples
        self.p_min = float("inf")
        self.s_min = float("inf")
        self.warning_level = warning_level
        self.drift_level = drift_level

    def update(self, prediction_is_correct):
        """
        Updates the detector with a new prediction result.
        :param prediction_is_correct: bool (True if correct, False if error)
        :return: "DRIFT", "WARNING", or "NORMAL"
        """
        self.n += 1
        error = 0 if prediction_is_correct else 1

        # Current error rate
        p = error / self.n if self.n == 1 else (error + (self.n - 1) * self.p_min) / self.n
        s = math.sqrt(p * (1 - p) / self.n)

        # Initialize min values
        if self.n == 1:
            self.p_min, self.s_min = p, s

        # Update if new minima
        if p + s < self.p_min + self.s_min:
            self.p_min, self.s_min = p, s

        # Drift detection
        if p + s > self.p_min + self.drift_level * self.s_min:
            return "DRIFT"
        elif p + s > self.p_min + self.warning_level * self.s_min:
            return "WARNING"
        else:
            return "NORMAL"


# 🔹 Exemplo de uso
if __name__ == "__main__":
    import random

    ddm = DDM()
    results = []

    # Simulação: primeiras 100 previsões corretas, depois começam erros
    for i in range(200):
        if i < 100:
            prediction_correct = True
        else:
            prediction_correct = random.random() > 0.6  # mais erros após 100

        state = ddm.update(prediction_correct)
        results.append(state)
        print(f"Step {i+1}: {state}")
