% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
% 
% This file is part of Handbook of Biometric Anti-Spoofing 2.

run_all_train
run_all_dev
run_all_eval

asvtrain = imageDatastore('train_features');
fileID = fopen('ASVspoof2017_train.trn.txt');
protocol = textscan(fileID,'%s%s%s%s%s%s%s');
labels = protocol{2};
asvtrain.Labels = labels;
asvtrain.Labels = categorical(asvtrain.Labels);
save("asvtrain.mat","asvtrain")

asvevalV2 = imageDatastore('eval_features');
fileID = fopen('ASVspoof2017_eval_v2_key.trl.txt');
protocol = textscan(fileID, '%s%s');
labels = protocol{2};
asvevalV2.Labels = labels;
asvevalV2.Labels = categorical(asvevalV2.Labels);
save("asvevalV2.mat","asvevalV2")

asvdev = imageDatastore('dev_features');
fileID = fopen('ASVspoof2017_dev.trl');
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
labels = protocol{2};
asvdev.Labels = labels;
asvdev.Labels = categorical(asvdev.Labels);
save("asvdev.mat","asvdev")
