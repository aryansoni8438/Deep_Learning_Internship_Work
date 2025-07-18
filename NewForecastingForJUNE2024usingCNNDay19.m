% Improved Normalization 0 to 1 range
[pn, ps] = mapminmax(p', 0, 1);  % Normalize input training data
[tn, ts] = mapminmax(t', 0, 1);  % Normalize target training data

XTrain = pn;
YTrain = tn;

numFeatures = size(XTrain, 1); % Number of inputs
numResponses = size(YTrain, 1); % Number of outputs

% Define CNN model layers
layers = [
    sequenceInputLayer(numFeatures, "Name", "input")
    convolution1dLayer(5, 128, "Padding", "same", "Name", "conv1") % Increased filter size and number
    batchNormalizationLayer("Name", "batchnorm1")
    reluLayer("Name", "relu1")
    dropoutLayer(0.3, "Name", "drop1") % Reduced dropout rate
    convolution1dLayer(5, 256, "Padding", "same", "Name", "conv2") % Increased filter size and number
    batchNormalizationLayer("Name", "batchnorm2")
    reluLayer("Name", "relu2")
    dropoutLayer(0.3, "Name", "drop2")
    convolution1dLayer(5, 512, "Padding", "same", "Name", "conv3") % Added another convolutional layer
    batchNormalizationLayer("Name", "batchnorm3")
    reluLayer("Name", "relu3")
    dropoutLayer(0.3, "Name", "drop3")
    fullyConnectedLayer(1024, "Name", "fc1") % Added fully connected layer
    reluLayer("Name", "relu4")
    dropoutLayer(0.3, "Name", "drop4")
    fullyConnectedLayer(numResponses, "Name", "fc2")
    regressionLayer("Name", "regressionoutput")];

% Optimized training options
options = trainingOptions("adam", ...
    "GradientThreshold", 1, ...
    "InitialLearnRate", 0.0001, ... % Reduced learning rate
    'MaxEpochs', 4000, ... % Increased number of epochs
    'MiniBatchSize', 64, ... % Adjusted batch size
    'SequenceLength', 'longest', ...
    'Epsilon', 1e-8, ...
    'L2Regularization', 0.001, ...
    "Shuffle", "every-epoch", ...
    'GradientDecayFactor', 0.9, ...
    'SquaredGradientDecayFactor', 0.999, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ... % Adjusted drop factor
    'LearnRateDropPeriod', 20, ... % Adjusted drop period
    'ResetInputNormalization', false, ... % Keep normalization consistent
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5); % Early stopping

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

  % Corrected variable name for denormalized prediction
error_var = Ytest'- YPred2new;
Et = Ytest' - New_V;


% Testing result plot
figure;
subplot(2, 1, 1);
plot(Ytest', 'b');
hold on;
plot(YPred2new, 'r');  % Corrected variable name
hold off;
legend("Actual", "Forecast");
xlabel("Data Points");
ylabel("Load Demand in MW");
title("Actual vs Forecast load by CNN (Target)");

subplot(2, 1, 2);
plot(Et);  % Corrected variable name
legend("Error");
xlabel("Data Points");
ylabel("Error");

% Calculate error metrics
RMSEtest = sqrt(mean(Et.^2));  % Corrected variable name
maetest = mean(abs(Et));  % Corrected variable name
mapetest = mean(abs(Et./ Ytest')) * 100;  % Corrected variable name

% Display error metrics
fprintf('Test RMSE: %.4f\n', RMSEtest);
fprintf('Test MAE: %.4f\n', maetest);
fprintf('Test MAPE: %.4f%%\n', mapetest);
