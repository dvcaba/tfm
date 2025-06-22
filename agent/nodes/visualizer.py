# nodes/visualizer.py
from agent.utils.loader import get_conf_matrix_path

def get_confusion_matrix():
    return {"confusion_matrix_path": get_conf_matrix_path()}
