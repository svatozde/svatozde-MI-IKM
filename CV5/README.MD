In fifth task i used only knns as ensemble methods,
and their f1 scores as weights here are the example results, the coloring of the areas is achived by calculating PCA
for the input datapoints, and then transforming 2d meshgrid back into dimensinon of the dataset the colorinf it then 
done be using classification on this calculated datapoint, this was posible only for datasets  
data_x_tst.csv and data_x_tr.csv dimension of the other datasets its to hi to provide nice visualization in 2d. :(


![ensembled clasifier](https://github.com/svatozde/svatozde-MI-IKM/blob/master/CV5/figs/showcase.png)

its possible to see that ensembling many models leads to significant improvement,
 since the f1 score of whole model was 1.0 (in the case on the visualization).

