# USAI10K-Model-Finetune
A repository for continuing the training of an EfficientNetB5 single-task model (89% accuracy) on the USAI10K dataset. It targets 15 disease classes via the prediction_layer and optimizes performance through a 2-phase workflow: freezing the backbone for classification head warm-up, followed by fine-tuning with selective block 4-7 unfreezing.
