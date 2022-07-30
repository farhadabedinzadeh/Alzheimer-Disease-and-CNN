clear;close;clc
%% '================ Written by Farhad AbedinZadeh ================'
%                                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%%
%Kaggle Dataset
% https://www.kaggle.com/tourist55/alzheimers-dataset-4-class-of-images?select=Alzheimer_s+Dataset

imds = imageDatastore('train', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%resizing
inputSize = [227 227];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imshow(preview(imds));
%%convert gray to rgb
imds.ReadFcn = @(loc)cat(3,imread(loc),imread(loc),imread(loc));
% imshow(preview(imds));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end
%% Load Pretrained Network
% Load the pretrained AlexNet neural network
net = alexnet;
%% 
analyzeNetwork(net)
%% 
% The first layer, the image input layer, requires input images of size 227-by-227-by-3, 
% where 3 is the number of color channels. 

inputSize = net.Layers(1).InputSize
%% Replace Final Layers
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels))
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%% Train Network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%% 
% To automatically resize the validation images without performing further data 
% augmentation, use an augmented image datastore without specifying any additional 
% preprocessing operations.

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
%% 
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augimdsTrain,layers,options);
%% Classify Validation Images
% Classify the validation images using the fine-tuned network.

[YPred,scores] = classify(netTransfer,augimdsValidation);
%% 
% Display four sample validation images with their predicted labels.

idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end
%% 
% Calculate the classification accuracy on the validation set. Accuracy is the 
% fraction of labels that the network predicts correctly.

YValidation = imdsValidation.Labels;
