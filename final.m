train = readtable("exoTrain.csv");
test  = readtable("exoTest.csv"); 
data = [train;test];
datadata = data(:,2:3198);
label = data(:,1);

writetable(datadata, 'data.csv');
writetable(label, 'label.csv');