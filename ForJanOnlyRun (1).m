data = TenJuneWork;
combined_data = mean(data, 2);
forecast_horizon =676
Mdl = arima('Constant',0,'D',1,'Seasonality',96, 'MALags', 1:3) ; 
try
    EstMdl = estimate(Mdl, combined_data);
catch ME
    disp(ME.message); 
    error('Unable to estimate the SARIMA model. Please adjust the model parameters.');
end
[New_Var, ~] = forecast(EstMdl, forecast_horizon, 'Y0', combined_data);
figure;
plot(1:length(combined_data), combined_data, 'b', 'LineWidth', 2);
hold on;
plot(length(combined_data)+(1:forecast_horizon), New_Var, 'r', 'LineWidth', 2);
hold off;
legend('Original Combined Signal', 'Forecasted Signal');
xlabel('Time');
ylabel('Value');
title('Original Combined Signal vs Forecasted Signal');
grid on

%figure
%error=New_Var-Only2024Jan
%subplot(2,1,1); plot(Only2024Jan); hold on ; plot(New_Var);
%legend("actual","forecasted");
 %subplot(2,1,2);plot(error);

 %mae=mean(abs(error));
 %mape=mean(abs(error./Only2024Jan)*100);