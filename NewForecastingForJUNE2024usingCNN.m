% Improved Normalization 0 to 1 range
[pn, ps] = mapminmax(p', 0, 1);
[tn, ts] = mapminmax(t', 0, 1);

XTrain = pn;
YTrain = tn;

numFeatures = size(XTrain, 1); % Number of inputs
numResponses = size(YTrain, 1); % Number of outputs

% Define CNN model layers
layers = [
    sequenceInputLayer(numFeatures, "Name", "input")
    convolution1dLayer(3, 64, "Padding", "same", "Name", "conv1")
    batchNormalizationLayer("Name", "batchnorm1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.5, "Name", "drop1")
    convolution1dLayer(3, 128, "Padding", "same", "Name", "conv2")
    batchNormalizationLayer("Name", "batchnorm2")
    reluLayer("Name", "relu2")
    dropoutLayer(0.5, "Name", "drop2")
    fullyConnectedLayer(numResponses, "Name", "fc")
    regressionLayer("Name", "regressionoutput")];

% Optimized training options
options = trainingOptions("adam", ...
    "GradientThreshold", 1, ...
    "InitialLearnRate", 0.001, ...
    'MaxEpochs', 3000, ...
    'SequenceLength', 'longest', ...
    'Epsilon', 1e-8, ...
    'L2Regularization', 0.001, ...
    "Shuffle", "every-epoch", ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.999, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 50, ...
    'ResetInputNormalization', true, ...
    "Plots", "training-progress");

% Training the CNN network
net = trainNetwork(XTrain, YTrain, layers, options);
an = predict(net, pn);

% Denormalizing the trained result
a = mapminmax('reverse', an, ts);

error = t' - a; % 'a' is trained forecasted result and 't' is original training target

% Training result plot
figure;
subplot(2, 1, 1);
plot(a, 'r');
hold on;
plot(t, 'b');
legend("Forecast for JUNE 2024");
title("Actual vs Forecast load by CNN (Train)");
errortrain = t' - a;
subplot(2, 1, 2);
plot(errortrain);
legend("Error");

figure;
histogram(errortrain);
title('Training Error Histogram');

% Testing
XTestn = mapminmax('apply', Xtest', ps); % Applying same structure of normalization on new testing dataset
YPrednew = predict(net, XTestn); % Prediction
YPred1new = double(YPrednew);
YPred2new = mapminmax('reverse', YPred1new, ts); % Denormalization of testing result

errortest = Ytest' - YPred2new;

% Testing result plot
figure;
subplot(2, 1, 1);
plot(Ytest', 'b');
hold on;
plot(YPred2new, 'r');
hold off;
legend("Actual", "Forecast");
xlabel("Data Points");
ylabel("Load Demand in MW");
title("Actual vs Forecast load by CNN (Target)");

subplot(2, 1, 2);
plot(errortest);
legend("Error");
xlabel("Data Points");
ylabel("Error");

% Calculate error metrics
RMSEtest = sqrt(mean(errortest.^2));
maetest = mean(abs(errortest));
mapetest = mean(abs(errortest ./ Ytest')) * 100;
