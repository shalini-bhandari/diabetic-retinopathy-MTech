## Experiment ID:Exp01_Intitial_Baseline
    Hypothesis: standard transfer learning mode, MobileNetV2, can achieve reasonable baseline on small balanced subset of data.
    Configuration:
        Model: MobileNetV2
        Dataset: Small, balanced subset (SAMPLES_PER_CLASS = 250).
        Training: Simple, single-stage training for 15 epochs.
        Preprocessing: Incorrect (rescale = 1./255.).
    Results:
        final validation accuracy : around 
    Observations & Notes:

        The model struggled significantly, with performance barely above random chance.

        The low accuracy was attributed to several factors: a very small dataset, incorrect image preprocessing for the pre-trained model, and a simplistic training strategy. This established the need for a more robust, full-scale approach.

## Experiment ID: Exp02_Aggressive_FineTuning
    Hypothesis: Using a full dataset, class weights, and a proper two-stage fine-tuning process with EfficientNetB0 will significantly improve performance.

    Configuration:

        Model: EfficientNetB0
        Dataset: Full, original dataset.
        Training: Two-stage fine-tuning.
        Fine-Tuning Strategy: Aggressive (unfreezing from layer 100, fine_tune_at = 100).
        Preprocessing: Correct (applications.efficientnet.preprocess_input).

    Results:   
        Peak Validation Accuracy (End of Stage 1): 0.7309
        Final Validation Accuracy (After Fine-Tuning): 0.7135 (Performance decreased).

    Observations & Notes:

        The feature extraction stage (Stage 1) was highly successful, establishing a strong baseline of ~73% accuracy.
        However, the fine-tuning stage was unstable and harmed the model's performance. This demonstrated that unfreezing too many layers at once can corrupt the pre-trained weights, a phenomenon known as catastrophic forgetting.
        

## Experiment ID: Exp03_Cautious_FineTuning
1. Unfreeze fewer layers: instead of unfreezing from layer 100, we will only unfreeze the layers at the very end of the network. UNfreezing from 200 layers is much safer.
2. increase fine tuning epochs: Give the model more time to adapt at the very low learning rate. Increasing the fine-tuning epochs from 10 to 15 or 20 can be beneficial.

Category according to the widely accepted Landis and Koch scale:
0.0 – 0.20: Slight agreement
0.21 – 0.40: Fair agreement
0.41 – 0.60: Moderate agreement
0.61 – 0.80: Substantial agreement 
0.81 – 1.00: Almost perfect agreement

    Hypothesis: A more cautious fine-tuning strategy (unfreezing fewer layers) will allow the EfficientNetB0 model to adapt successfully without destabilizing.

    Configuration:

        Model: EfficientNetB0
        Dataset: Full, original dataset.
        Training: Two-stage fine-tuning.
        Fine-Tuning Strategy: Cautious (unfreezing from layer 200, fine_tune_at = 200).

    Results:

        Peak Validation Accuracy (End of Stage 1): ~0.73
        Final Validation Accuracy (After Fine-Tuning): 0.7674
        Final Test Set Cohen's Kappa Score: 0.6303

    Observations & Notes:

        This approach was successful. The cautious fine-tuning allowed the model to build upon the strong baseline from Stage 1, resulting in a significant performance increase.
        The final Kappa score of 0.6303 indicates "Substantial Agreement," proving this is a robust and effective model. This became our final, successful baseline.

## Experiment ID: Exp0 Progressive Unfreezing on B0

## Experiment ID: Exp0 B1_model
    Model: EfficientNetB1 

## Experiment ID: Exp0 B2_model
    Model: EfficientNetB2 


## Experiment ID: Exp0 Progressive Unfreezing on B2