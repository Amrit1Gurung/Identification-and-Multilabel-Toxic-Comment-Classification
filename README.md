# Identification-and-Multilabel-Toxic-Comment-Classification  
Discussing things you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments. So classifying the user comments or opinions for toxicity is important to make effective and abuse free online social platforms. Here we experimented with various models to classify online comments/text to various toxic categories using Machine learning and Deep learning algorithms.  
## Model Comparison  
A comparison of few models used in this repo(Deep learning model not included here but was best performer qqually to SVM). 
| Category      | Acc NB             | Acc LogReg         | Acc SVM            | SVM advantage to NB   | SVM advantage to LogReg |
|---------------|--------------------|--------------------|--------------------|-----------------------|-------------------------|
| toxic         | 0.9499295002349992 | 0.956509478301739  | 0.9586088046373179 | 0.008679304402318677  | 0.002099326335578855    |
| severe_toxic  | 0.990474698417672  | 0.9905060316465611 | 0.9907253642487859 | 0.0002506658311138832 | 0.00021933260222473105  |
| obscene       | 0.9721760927463575 | 0.977659407801974  | 0.9792887357042144 | 0.007112642957856852  | 0.0016293279022403517   |
| threat        | 0.9976813410621964 | 0.9977753407488642 | 0.9977753407488642 | 9.399968666778946e-05 | 0.0                     |
| insult        | 0.9665674447751841 | 0.9705467648441172 | 0.9707347642174526 | 0.004167319442268558  | 0.0001879993733354679   |
| identity_hate | 0.9911013629954567 | 0.9918533604887984 | 0.991947360175466  | 0.000845997180009328  | 9.399968666767844e-05   |

