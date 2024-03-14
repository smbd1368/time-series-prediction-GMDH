clc;
clear;
close all;

%% Load Data

x=load('data.mat');
x=x.x;
Inputs=x([1:1000],[1,2,3,4])';
Targets=x([1:1000],[5])';
Delays = [5 10 15 20 25];

%%[Inputs, Targets] = CreateTimeSeriesData(x,Delays);

nData = size(Inputs,2);
Perm = randperm(nData);

% Train Data
pTrain = 0.9;
nTrainData = round(pTrain*nData);
TrainInd = Perm(1:nTrainData);
TrainInputs = Inputs(:,TrainInd);
TrainTargets = Targets(:,TrainInd);

% Test Data
pTest = 1 - pTrain;
nTestData = nData - nTrainData;
TestInd = Perm(nTrainData+1:end);
TestInputs = Inputs(:,TestInd);
TestTargets = Targets(:,TestInd);

%% Create and Train GMDH Network

params.MaxLayerNeurons = 30;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 5;          % Maximum Number of Layers
params.alpha = 0;              % Selection Pressure
params.pTrain = 0.7;           % Train Ratio
gmdh = GMDH(params, TrainInputs, TrainTargets);

%% Evaluate GMDH Network

Outputs = ApplyGMDH(gmdh, Inputs);
TrainOutputs = Outputs(:,TrainInd);
TestOutputs = Outputs(:,TestInd);

%% Show Results

figure;
PlotResults(TrainTargets, TrainOutputs, 'Train Data');

figure;
PlotResults(TestTargets, TestOutputs, 'Test Data');

figure;
PlotResults(Targets, Outputs, 'All Data');

figure;
plotregression(TrainTargets, TrainOutputs, 'Train Data', ...
               TestTargets, TestOutputs, 'TestData', ...
               Targets, Outputs, 'All Data');
           
