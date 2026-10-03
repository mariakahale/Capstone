clear;
clc;

scriptFolder = fileparts(mfilename('fullpath'));
if ~isempty(scriptFolder)
    cd(scriptFolder);
end

% Reconstructed code for Figure 4: age-dependent pole fragility curves.
% The model inputs and deterioration equations are taken from the archived
% Monte Carlo script. The three plotted cases are new, 30-year, and 60-year
% poles. Units are SI unless otherwise noted.

rng(1, 'twister');                 % Reproducible Monte Carlo results
n = 500000;                        % Number of Monte Carlo samples
V = (20:0.1:90)';                  % Extreme wind speed plotted in Figure 4 (m/s)

% Random wind-load parameters
Kzp = normrnd(0.951, 0.05706, n, 1);
Kzw = normrnd(1.024, 0.06144, n, 1);
Gp  = normrnd(0.948, 0.10428, n, 1);
Gw  = normrnd(0.801, 0.08811, n, 1);
Cfp = normrnd(0.9,   0.108,   n, 1);
Cfw = normrnd(1.0,   0.12,    n, 1);
Ap  = normrnd(2.66,  0.1596,  n, 1);
Aw  = normrnd(5.16,  0.3098,  n, 1);
Hp  = normrnd(11.7,  0.351,   n, 1);
Hw  = normrnd(11.1,  0.333,   n, 1);

Kp = 0.613 .* Kzp .* Gp .* Cfp .* Ap;
Kw = 0.613 .* Kzw .* Gw .* Cfw .* Aw;

% Pole and gravity parameters
D = 0.28;                          % Pole diameter (m)
Gi = 7766 * 0.09144;               % Gravity load used in archived code (N)
gravityStress = Gi * 3 / ((D/2)^2 * pi); % Pa

% Because wind-induced stress is proportional to V^2, compute the critical
% wind speed of every Monte Carlo sample once. The empirical CDF of those
% critical speeds is identical to checking failure separately at each V,
% but it is much faster than the original nested loops.
windStressCoefficient = 32 .* (0.5 .* Kp .* Hp + Kw .* Hw) ./ (pi .* D^3);

% New-pole resistance distribution from the archived script (kPa -> Pa)
strengthNew = lognrnd(10.8486, 0.16879, n, 1) .* 1000;

% Aged-pole resistance distributions
strength30 = agedPoleStrength(30, n) .* 1000;
strength60 = agedPoleStrength(60, n) .* 1000;

fragilityNew = fragilityFromCriticalSpeed( ...
    strengthNew, gravityStress, windStressCoefficient, V);
fragility30 = fragilityFromCriticalSpeed( ...
    strength30, gravityStress, windStressCoefficient, V);
fragility60 = fragilityFromCriticalSpeed( ...
    strength60, gravityStress, windStressCoefficient, V);

% Recreate the Figure 4 presentation
figure('Color', 'w');
plot(V, fragilityNew, 'r-',  'LineWidth', 1.5);
hold on;
plot(V, fragility30,  'k-.', 'LineWidth', 1.5);
plot(V, fragility60,  'b--', 'LineWidth', 1.5);
hold off;

xlim([20 90]);
ylim([0 1]);
xticks(20:5:90);
yticks(0:0.1:1);
xlabel('Extreme wind speed (m/s)');
ylabel('Probability of failure');
legend('New poles', '30-yr poles', '60-yr poles', ...
    'Location', 'northwest');
box on;
set(gca, 'FontName', 'Arial', 'FontSize', 12, 'LineWidth', 1);

% Export the reconstructed curves for checking or sharing.
curveData = table(V, fragilityNew, fragility30, fragility60, ...
    'VariableNames', {'WindSpeed_mps', 'NewPoles', ...
    'Poles30yr', 'Poles60yr'});
writetable(curveData, 'figure4_fragility_curves.csv');
exportgraphics(gcf, 'reconstructed_figure4.png', 'Resolution', 300);
savefig(gcf, 'reconstructed_figure4.fig');


function strengthKPa = agedPoleStrength(ageYears, n)
% Reconstruct the age-dependent lognormal resistance model.

meanStrengthKPa = 52200 .* (1 - ...
    (0.014418 .* ageYears - 0.10683) .* ...
    (0.00013 .* ageYears.^1.846));

% In the original loop, age = 5*(t+1) and
% Rcov = 0.17 + 0.01667*t + 0.01667.
covStrength = 0.17 + 0.01667 .* (ageYears ./ 5);

muLog = log(meanStrengthKPa.^2 ./ ...
    sqrt((covStrength .* meanStrengthKPa).^2 + meanStrengthKPa.^2));
sigmaLog = sqrt(log(covStrength.^2 + 1));

strengthKPa = lognrnd(muLog, sigmaLog, n, 1);
end


function probabilityFailure = fragilityFromCriticalSpeed( ...
    strengthPa, gravityStress, windStressCoefficient, windSpeed)
% Convert each realization into its critical failure wind speed, then
% evaluate the empirical CDF at the requested wind speeds.

n = numel(strengthPa);
requiredWindStress = strengthPa - gravityStress;
criticalWindSpeed = inf(n, 1);

alreadyFailed = requiredWindStress <= 0;
criticalWindSpeed(alreadyFailed) = 0;

valid = requiredWindStress > 0 & windStressCoefficient > 0;
criticalWindSpeed(valid) = sqrt( ...
    requiredWindStress(valid) ./ windStressCoefficient(valid));

edges = [-inf; windSpeed; inf];
counts = histcounts(criticalWindSpeed, edges);
probabilityFailure = cumsum(counts(1:end-1)).' ./ n;
end
