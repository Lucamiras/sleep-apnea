MEAN = [0.36600792, 0.12097905, 0.40580472]
STD = [0.23583658, 0.10515513, 0.102424406]
SIZE = (224, 224)
TARGET_SR = 48_000
CHUNK_DURATION = 30
N_MELS = 128
AHI = {
    range(0, 5): "Healthy",
    range(5, 15): "Mild",
    range(15, 30): "Moderate",
    range(30, 120): "Severe"
}
CLASSES = {
    0: "No Apnea",
    1: "Hypopnea",
    2: "Obstructive Apnea (OSA)"
}
