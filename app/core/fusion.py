import numpy as np

STRATEGIES = [
    "OR (Maximum Coverage)",
    "AND (Maximum Precision)",
    "Weighted Voting",
    "Tiered (Anomaly screens, Rules validate)",
]

DESCRIPTIONS = {
    "OR (Maximum Coverage)": "Flag if **either** engine detects it. Highest recall, most false positives. Best for high-security environments where missing an attack is unacceptable.",
    "AND (Maximum Precision)": "Flag only if **both** engines agree. Fewest false positives but misses attacks caught by only one engine. Best for small teams who cannot handle high alert volumes.",
    "Weighted Voting": "Each engine gets a configurable weight against a threshold. Tuneable middle ground — requires calibration. Best for mature SOCs with resources to fine-tune parameters.",
    "Tiered (Anomaly screens, Rules validate)": "Anomaly engine casts a wide net, rule engine validates. Mimics real SOC workflow, reduces alert fatigue. Most realistic production deployment model.",
}


def or_fusion(a_preds, r_preds):
    return np.maximum(a_preds, r_preds)

def and_fusion(a_preds, r_preds):
    return np.minimum(a_preds, r_preds)

def weighted_fusion(a_preds, r_preds, a_weight=0.6, r_weight=0.4, threshold=0.5):
    combined = (a_preds * a_weight) + (r_preds * r_weight)
    return (combined >= threshold).astype(int)

def tiered_fusion(a_preds, r_preds):
    result = np.zeros_like(a_preds)
    both_agree = (a_preds == 1) & (r_preds == 1)
    result[both_agree] = 1
    return result

def run_fusion(a_preds, r_preds, strategy, a_weight=0.6, r_weight=0.4, threshold=0.5):
    if strategy == "OR (Maximum Coverage)":
        return or_fusion(a_preds, r_preds)
    elif strategy == "AND (Maximum Precision)":
        return and_fusion(a_preds, r_preds)
    elif strategy == "Weighted Voting":
        return weighted_fusion(a_preds, r_preds, a_weight, r_weight, threshold)
    elif strategy == "Tiered (Anomaly screens, Rules validate)":
        return tiered_fusion(a_preds, r_preds)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")