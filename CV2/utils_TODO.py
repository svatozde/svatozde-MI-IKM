import numpy as np
import pandas as pd
import matplotlib as plt

class Utils:
    
    def __init__(self, gt):
        self.gt = gt
        
    def accuracy(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)

        return (tp+tn)/(tp + tn + fp + fn)
        pass
    
    def false_positives(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return fp
    
    def false_negatives(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return fn

    
    def recall(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return tp / (tp + tn)
    
    def precision(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return tp / (tp + fp)
        
    def false_negative_rate(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return fn / (fn + tp)
   
    def false_positive_rate(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return fp / (fp + tn)
    
    def true_positives(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return tp
             
    def true_positive_rate(self, y_pred):
        tp, tn, fp, fn = self.count(self.gt, y_pred)
        return tp / (tp + fn)

    def roc(self, y_pred):

      

        # predict probabilities
        lr_probs = model.predict_proba(testX)
        # keep probabilities for the positive outcome only
        lr_probs = lr_probs[:, 1]
        # calculate scores
        ns_auc = roc_auc_score(testy, ns_probs)
        lr_auc = roc_auc_score(testy, lr_probs)
        # summarize scores
        print('No Skill: ROC AUC=%.3f' % (ns_auc))
        print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
        ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
        lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
        # plot the roc curve for the model
        pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
        # axis labels
        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        # show the legend
        pyplot.legend()
        # show the plot

    def count(self, gt, y_pred):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0, len(y_pred)):
            if gt[i] == 0:
                if y_pred[i] == 0:
                    tn= tn + 1
                else:
                    fn = fn + 1
            else:
                if y_pred[i] == 1:
                    tp = tp + 1
                else:
                    fp = fp + 1

        return tp, tn, fp, fn


def main():
    gt = pd.read_csv('C:\skola\MI-IKM\CV2\data\csv\dataset1_y_tr.csv')
    gt_v =   [ v[0]  for v in gt.iloc[:,0:].values ]


    y_pred = pd.read_csv('C:\skola\MI-IKM\CV2\data\csv\dataset1_y_tst.csv')
    y_pred_v = [ v[0]  for v in y_pred.iloc[:,0:].values ]

    utils = Utils(gt_v)
    print("true_positives:" +  str(utils.true_positives(y_pred_v)))
    print("false_positives:" + str(utils.false_positives(y_pred_v)))
    print("false_negative_rate:" + str(utils.false_negative_rate(y_pred_v)))
    print("precision:" + str(utils.precision(y_pred_v)))
    print("recall:" + str(utils.recall(y_pred_v)))
    print("accuracy:" + str(utils.accuracy(y_pred_v)))



if __name__ == '__main__':
   main();
