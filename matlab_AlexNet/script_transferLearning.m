% Copyright (c) 2018 Queensland University of Technology, Idiap Research Institute (http://www.idiap.ch/)
% Written by Ivan Himawan <i.himawan@qut.edu.au>,
%
% This file is part of Handbook of Biometric Anti-Spoofing 2.

% run on GPU, set CUDA_VISIBLE_DEVICE=0
% rng('default');
% rng(1);
% addpath(genpath('/home/himawan/alexnet'));

% 1. Perform transfer learning
net = alexnet

load('asvtrain.mat')

trainingImages.Labels = asvtrain.Labels;

numClasses = numel(categories(trainingImages.Labels));

layersTransfer = net.Layers(1:end-3);

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

miniBatchSize=32;

options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4,...
    'Verbose',true);

netTransfer = trainNetwork(asvtrain,layers,options);

% 2. Classify evaluation data
load('asvevalV2.mat');

layer = 'softmax';
testingFeatures_trans = activations(netTransfer, asvevalV2, layer);

evalProtocolFile = 'ASVspoof2017_eval_v2_key.trl.txt';
% read evaluation protocol
fileID = fopen(evalProtocolFile);
protocol = textscan(fileID, '%s%s');
fclose(fileID);

% get file and label lists
filelist = protocol{1};
labels = protocol{2};

addpath(genpath('bosaris_toolkit'));
scores = testingFeatures_trans;

% compute performance
scores = log(scores+eps)-log(1-scores+eps);

[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100; 
fprintf('EER is %.2f\n', EER);

% --- Classify development data
clear scores;
fprintf('Now dev set ...');

load('asvdev.mat');
layer = 'softmax';
asvdev_scores = activations(netTransfer,asvdev,layer);

devProtocolFile = 'ASVspoof2017_dev.trl';
% read development protocol
fileID = fopen(devProtocolFile);
protocol = textscan(fileID, '%s%s%s%s%s%s%s');
fclose(fileID)                                                                                      
% get file and label lists
filelist = protocol{1};
labels = protocol{2};

scores = asvdev_scores(:,1);
scores = log(scores+eps)-log(1-scores+eps);

[Pmiss,Pfa] = rocch(scores(strcmp(labels,'genuine')),scores(strcmp(labels,'spoof')));
EER = rocch2eer(Pmiss,Pfa) * 100;
fprintf('EER is %.2f\n', EER);

