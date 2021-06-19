# Predicting glaucoma progression requiring surgery using clinical free-text notes and transfer learning with transformers
Housing code and project files related to using BERT-based models to predict whether a patient will need glaucoma surgery or not, with data from electronic health records.

# Abstract 
Purpose: We evaluate transfer learning from four different BERT-based models to use clinical notes to predict glaucoma progression requiring surgery.
Methods: Ophthalmology clinical notes of 4512 glaucoma patients at a single center from 2008-2020 were identified from electronic health records (EHR). 748 patients had glaucoma surgery. Pre-trained BERT-based models were fine-tuned on free-text clinical notes for the task of predicting which patients would require surgery. Models were compared by the area under the receiver operating characteristic curve (AUROC).
Results: BERT had the best AUROC 73.4%, followed by RoBERTa with AUROC 72.4%, DistilBERT with AUROC 70.2%, and BioBERT with AUROC 70.1%. All models outperformed an ophthalmologist’s review of clinical notes (F1 29.9%).
Conclusion: Transformers with transfer learning can predict whether glaucoma progression requires surgery. Future work can focus on improving model performance, potentially by integrating structured or imaging data. 

