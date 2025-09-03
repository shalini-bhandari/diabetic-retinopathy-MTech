import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from src import config
from src.data_loader import create_data_generators

def plot_confusion_matrix(y_true, y_pred, class_labels):
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'))
    plt.show()

def main():
    _, _, test_generator = create_data_generators()

    print("\nEvaluating on Test Set")
    best_model = tf.keras.models.load_model(config.BEST_MODEL_PATH)
    
    predictions = best_model.predict(test_generator, steps=len(test_generator))
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_labels)
    
    print(f"Cohen's Kappa Score: {kappa:.4f}")
    print("\nClassification Report:")
    print(report)

    plot_confusion_matrix(y_true, y_pred, class_labels)

if __name__ == '__main__':
    main()