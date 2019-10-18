import importlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, cohen_kappa_score


x, y = np.load("x_test.npy"), np.load("y_test.npy")
# Here x is a NumPy array. On the actual exam it will be a list of paths.
# For the sake of this example, on the actual leaderboard we will use your nicknames
usernames = ["pedrouriar", "ajafari"]

scores, f1_scores, cohen_kappa_scores, nicknames, error_messages = [], [], [], [], []
for idx, username in enumerate(usernames):
    try:
        predict_script = importlib.import_module("predict_{}".format(username))
        y_pred, model = predict_script.predict(x)
        f1_scores.append(f1_score(y, y_pred, average="macro"))
        cohen_kappa_scores.append(cohen_kappa_score(y, y_pred))
        scores.append((f1_scores[-1] + cohen_kappa_scores[-1])/2)
        error_messages.append(None)
    except Exception as e:
        f1_scores.append(0); cohen_kappa_scores.append(0); scores.append(0);
        if "predict_" in str(e):
            error_messages.append("Missing predict script")
        else:
            error_messages.append(e)
    print(idx+1, "ouf of", len(usernames), "done")

leaderboard = pd.DataFrame({"Score": scores, "Nickname": usernames, "F1-score (macro)": f1_scores,
                            "Cohen's Kappa Score": cohen_kappa_scores, "Error": error_messages})
leaderboard.sort_values(by=["Score"], inplace=True, ascending=False)
leaderboard.to_excel("leaderboard.xlsx")
