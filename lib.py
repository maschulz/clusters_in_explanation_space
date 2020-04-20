import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_array, check_X_y


class ExRF:
    def __init__(self, **args):
        self.model = RandomForestClassifier(**args)

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x, y)

    def explain_shap(self, x):
        x = check_array(x)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(x)

        return shap_values[1]
