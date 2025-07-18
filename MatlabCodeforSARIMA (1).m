data = Input_Data;
combined_data = data;

forecast_horizon = 672 ; %for one week (96 X 7)
Mdl = arima('Constant',0,'D',1,'Seasonality',2976, 'MALags', 1:96);

try
    EstMdl = estimate(Mdl, combined_data);
catch ME
    disp(ME.message); 
    error('Unable to estimate the SARIMA model. Please adjust the model parameters.');
end

[Forecasted_Signal, ~] = forecast(EstMdl, forecast_horizon, 'Y0', combined_data);

figure;
plot(1:length(combined_data), combined_data, 'b', 'LineWidth', 2);
hold on;
plot(length(combined_data)+(1:forecast_horizon), Forecasted_Signal, 'r', 'LineWidth', 2);
hold off;
legend('Original Combined Signal', 'Forecasted Signal');
xlabel('Time');
ylabel('Value');
title('Original Combined Signal vs Forecasted Signal');
grid on;
