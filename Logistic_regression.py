from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

model_num = 50   # Number of models to make the averaged performance
coeff_1, coeff_2 = [], []
accu_p, accu_n = [], []
for i in tqdm(range(0,model_num)):
    np.random.seed(i)
    y_pred = []

    np.random.shuffle(label_p)     # Positive (Exp observed matched data)     [energy, occurrence]
    np.random.shuffle(label_n)     # Negative (Exp observed non-matched data) [energy, occurrence]

    train_1 = label_p[:600]+label_n[:600]
    test_1 = label_p[600:700]+label_n[600:700]

    train_label = np.concatenate([np.ones(600), np.zeros(600)])
    test_label = np.concatenate([np.ones(100),np.zeros(len(label_n[600:700]))])
    
    X_test = test_1
    y_test = test_label
    
    for a1,a2 in test_1:
        if a1<0.15:
            y_pred.append(1)
        elif a2>0.005:
            y_pred.append(1)
        else:
            y_pred.append(0)            
        
        
    correct,num_see = 0,100
    for yt,yp in zip(y_test[:num_see], y_pred[:num_see]):
        if int(yt) == yp:
            correct += 1
    accu_p.append(correct/num_see)

    correct,num_see = 0,100
    for yt,yp in zip(y_test[100:100+num_see], y_pred[100:100+num_see]):
        if int(yt) == yp:
            correct += 1

    accu_n.append(correct/num_see)
    
    

mean_p, mean_n = float(np.mean(accu_p)), float(np.mean(accu_n))
accu, prec = (mean_p+mean_n)/2, mean_p/(mean_p+1-mean_n)
recall = mean_p
f1_score = 2*(prec*recall)/(prec+recall)
print('TPR : %f, TNR : %f' %(mean_p, mean_n))
print('Accuracy : %f, Precision : %f, Recall : %f, F1_score : %f' %(accu,prec,recall,f1_score))