% Real-World Applications of Bifurcation Theory
% This script demonstrates bifurcations in practical engineering and scientific systems

clear; clc; close all;

warning('off', 'MATLAB:ode45:IntegrationTolNotMet');
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Events', @eventfun);

%% 1. STRUCTURAL ENGINEERING: EULER BUCKLING (Pitchfork Bifurcation)
% Column buckling under axial load: P_critical = π²EI/L²
% dx/dt = -dV/dx where V(x) = (1/2)kx² - (1/6)αx³ - Fx

figure('Name', 'Structural Engineering: Euler Buckling', 'Position', [100, 100, 1400, 500]);

% Parameters
E = 200e9;  % Young's modulus (Pa) - steel
I = 8.33e-6; % Moment of inertia (m⁴) - circular cross-section
L = 2;      % Column length (m)
P_critical = pi^2 * E * I / L^2;  % Critical buckling load

% Dimensionless load parameter
P_range = linspace(0, 2*P_critical, 1000);
lambda = P_range / P_critical;  % Dimensionless load

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
% Plot stable straight configuration for λ < 1
h1 = plot(lambda(lambda <= 1), zeros(size(lambda(lambda <= 1))), 'b-', 'LineWidth', 3); hold on;

% Plot unstable straight configuration for λ > 1
h2 = plot(lambda(lambda >= 1), zeros(size(lambda(lambda >= 1))), 'r--', 'LineWidth', 3);

% Plot stable buckled configurations for λ > 1
buckled_lambda = lambda(lambda >= 1);
buckled_w = sqrt(buckled_lambda - 1);
h3 = plot(buckled_lambda, buckled_w, 'b-', 'LineWidth', 2);
plot(buckled_lambda, -buckled_w, 'b-', 'LineWidth', 2);

% Bifurcation point
h4 = plot(1, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

xlabel('Dimensionless Load λ = P/P_{critical}');
ylabel('Dimensionless Deflection w');
title('Euler Buckling: Supercritical Pitchfork');
legend([h1, h2, h3, h4], {'Straight (Stable)', 'Straight (Unstable)', 'Buckled (Stable)', 'Bifurcation Point'}, 'Location', 'best');
grid on;

% Subplot 2: Energy landscape
subplot(1,3,2);
w_vals = linspace(-2, 2, 1000);
lambda_examples = [0.5, 1.0, 1.5];
colors = ['b', 'k', 'r'];

for i = 1:length(lambda_examples)
    lam = lambda_examples(i);
    % Potential energy: V(w) = (1/2)w² - (1/4)λw² + (1/6)w⁴
    V = 0.5*w_vals.^2 - 0.25*lam*w_vals.^2 + (1/6)*w_vals.^4;
    plot(w_vals, V, colors(i), 'LineWidth', 2); hold on;
end

xlabel('Dimensionless Deflection w');
ylabel('Potential Energy V');
title('Energy Landscape');
legend('λ = 0.5', 'λ = 1.0', 'λ = 1.5', 'Location', 'best');
grid on;

% Subplot 3: Load-deflection response
subplot(1,3,3);
% Simulate loading and unloading cycle
w_max = 1.5;
w_load = linspace(0, w_max, 100);
w_unload = linspace(w_max, 0, 100);

% Loading: P = P_critical * (1 + w²) for stable buckled state
P_load = P_critical * (1 + w_load.^2);
P_unload = P_critical * (1 + w_unload.^2);

plot(w_load, P_load/1e6, 'b-', 'LineWidth', 2); hold on;
plot(w_unload, P_unload/1e6, 'r--', 'LineWidth', 2);
plot([0, w_max], [P_critical/1e6, P_critical/1e6], 'k:', 'LineWidth', 1);

xlabel('Deflection w (dimensionless)');
ylabel('Applied Load P (MN)');
title('Load-Deflection Response (Hysteresis)');
legend('Loading', 'Unloading', 'Critical Load', 'Location', 'best');
grid on;

%% 1.5 STRUCTURAL DYNAMICS: TACOMA BRIDGE COLLAPSE (Hopf Bifurcation)
% Proper flutter model using flutter derivatives
% Coupled heaving-torsional motion with aerodynamic coupling
% Based on Theodorsen's theory and bridge aerodynamics

figure('Name', 'Structural Dynamics: Tacoma Bridge Flutter', 'Position', [125, 125, 1400, 500]);

% Bridge parameters (Tacoma Narrows Bridge)
m = 8900;       % Mass per unit length (kg/m)
I = 1.39e6;     % Moment of inertia per unit length (kg*m^2/m)
omega_h = 0.62; % Heaving frequency (rad/s) - vertical motion
omega_theta = 1.78; % Torsional frequency (rad/s) - twisting motion
zeta_h = 0.02;  % Heaving damping ratio
zeta_theta = 0.01; % Torsional damping ratio
B = 11.9;       % Half-width of bridge deck (m)

% Air properties
rho = 1.225;    % Air density (kg/m^3)

% Flutter derivatives (dimensionless) - based on bridge aerodynamics literature
% Negative values indicate destabilizing effects that can lead to flutter
H1_star = 0.2;    % Heaving added mass (positive)
H2_star = -0.8;   % Heaving aerodynamic damping (negative = destabilizing)
H3_star = 0.1;    % Heaving-torsional coupling
H4_star = -0.3;   % Heaving-torsional damping coupling
A1_star = -0.15;  % Torsional-heaving coupling (negative = destabilizing)
A2_star = -1.2;   % Torsional-heaving damping (negative = destabilizing)
A3_star = 0.05;   % Torsional added mass
A4_star = -0.6;   % Torsional aerodynamic damping (negative = destabilizing)

% Wind speed range
V_range = linspace(5, 50, 1000);  % m/s

% Analyze stability for each wind speed
max_real_part = zeros(size(V_range));
flutter_freq = zeros(size(V_range));

for i = 1:length(V_range)
    V = V_range(i);
    
    % Reduced velocity
    V_red_h = V / (omega_h * B);  % For heaving
    V_red_theta = V / (omega_theta * B);  % For torsional
    
    % Dynamic pressure
    q = 0.5 * rho * V^2;
    
    % Aerodynamic forces based on flutter derivatives
    % These depend on reduced velocity and create coupling between modes
    
    % Aerodynamic forces per unit span:
    % F_h = (1/2) * rho * V^2 * (2B) * [H1*(h_dot/V) + H2*(B*theta_dot/V) + H3*(h*omega_h/V) + H4*(B*theta*omega_theta/V)]
    % M_theta = (1/2) * rho * V^2 * (2B) * B * [A1*(h_dot/V) + A2*(B*theta_dot/V) + A3*(h*omega_h/V) + A4*(B*theta*omega_theta/V)]
    
    % Rewrite in terms of reduced frequency k = omega*B/V
    % For flutter analysis, we use the eigenvalue formulation
    
    % Generalized aerodynamic forces in matrix form
    % The key insight: flutter derivatives create velocity-dependent forces
    
    % Mass matrix terms (aerodynamic added mass)
    M_aero_hh = rho * B^2 * H1_star;
    M_aero_h_theta = rho * B^3 * H3_star;
    M_aero_theta_h = rho * B^3 * A1_star;
    M_aero_theta_theta = rho * B^4 * A3_star;
    
    % Damping matrix terms (aerodynamic damping)
    C_aero_hh = rho * V * B^2 * H2_star;
    C_aero_h_theta = rho * V * B^3 * H4_star;
    C_aero_theta_h = rho * V * B^3 * A2_star;
    C_aero_theta_theta = rho * V * B^4 * A4_star;
    
    % Total system matrices
    % Mass matrix: [M_struct + M_aero]
    M_total = [m + M_aero_hh,           M_aero_h_theta; ...
               M_aero_theta_h,          I + M_aero_theta_theta];
    
    % Damping matrix: [C_struct + C_aero]
    C_total = [2*zeta_h*omega_h*m + C_aero_hh,     C_aero_h_theta; ...
               C_aero_theta_h,                     2*zeta_theta*omega_theta*I + C_aero_theta_theta];
    
    % Stiffness matrix: [K_struct] (no aerodynamic stiffness)
    K_total = [m * omega_h^2,  0; ...
               0,              I * omega_theta^2];
    
    % State space form: x_dot = A*x where x = [h, h_dot, theta, theta_dot]
    % From: M*x_ddot + C*x_dot + K*x = 0
    % Rearrange: x_ddot = -M^(-1)*K*x - M^(-1)*C*x_dot
    
    try
        M_inv = inv(M_total);
        A = [zeros(2,2),        eye(2); ...
             -M_inv * K_total,  -M_inv * C_total];
        
        % Eigenvalues
        eigenvals = eig(A);
        
        % Find the eigenvalue with largest real part
        [max_real_part(i), idx] = max(real(eigenvals));
        
        % Flutter frequency (imaginary part of critical eigenvalue)
        flutter_freq(i) = abs(imag(eigenvals(idx))) / (2*pi);
        
    catch
        max_real_part(i) = NaN;
        flutter_freq(i) = NaN;
    end
end

% Find Hopf bifurcation point (where max real part crosses zero)
stable_idx = max_real_part < 0;
unstable_idx = max_real_part > 0;

% Find crossing point
crossing_idx = find(diff(sign(max_real_part)) > 0, 1);
if ~isempty(crossing_idx)
    % Interpolate to find exact crossing
    V1 = V_range(crossing_idx);
    V2 = V_range(crossing_idx + 1);
    lambda1 = max_real_part(crossing_idx);
    lambda2 = max_real_part(crossing_idx + 1);
    
    V_hopf = V1 - lambda1 * (V2 - V1) / (lambda2 - lambda1);
    
    % Interpolate flutter frequency at bifurcation
    freq_hopf = flutter_freq(crossing_idx) + ...
                (flutter_freq(crossing_idx + 1) - flutter_freq(crossing_idx)) * ...
                (V_hopf - V1) / (V2 - V1);
else
    V_hopf = NaN;
    freq_hopf = NaN;
end

% Subplot 1: Hopf bifurcation diagram
subplot(1,3,1);
% Plot stable and unstable regions
if any(stable_idx)
    h1 = plot(V_range(stable_idx), max_real_part(stable_idx), 'b-', 'LineWidth', 2); hold on;
end
if any(unstable_idx)
    h2 = plot(V_range(unstable_idx), max_real_part(unstable_idx), 'r-', 'LineWidth', 2); hold on;
end

% Zero line
h3 = plot(V_range, zeros(size(V_range)), 'k--', 'LineWidth', 1); hold on;

% Mark Hopf bifurcation point
if ~isnan(V_hopf)
    h4 = plot(V_hopf, 0, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
end

% Historical data point
h5 = plot(19.3, 0, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');

xlabel('Wind Speed V (m/s)');
ylabel('Real Part of Eigenvalue (1/s)');
title('Tacoma Bridge: Hopf Bifurcation Analysis');
ylim([-2, 2]);

% Create legend
legend_handles = [h3];
legend_labels = {'Neutral Stability'};
if any(stable_idx)
    legend_handles = [legend_handles, h1];
    legend_labels = [legend_labels, {'Stable (Damped)'}];
end
if any(unstable_idx)
    legend_handles = [legend_handles, h2];
    legend_labels = [legend_labels, {'Unstable (Flutter)'}];
end
if ~isnan(V_hopf)
    legend_handles = [legend_handles, h4];
    legend_labels = [legend_labels, {sprintf('Hopf Point (%.1f m/s)', V_hopf)}];
end
legend_handles = [legend_handles, h5];
legend_labels = [legend_labels, {'Historical (19.3 m/s)'}];

legend(legend_handles, legend_labels, 'Location', 'best');
grid on;

% Subplot 2: Flutter frequency
subplot(1,3,2);
plot(V_range, flutter_freq, 'r-', 'LineWidth', 2); hold on;

% Mark natural frequencies
plot(V_range, (omega_h/(2*pi))*ones(size(V_range)), 'b--', 'LineWidth', 1);
plot(V_range, (omega_theta/(2*pi))*ones(size(V_range)), 'g--', 'LineWidth', 1);

% Mark bifurcation point
if ~isnan(V_hopf)
    plot(V_hopf, freq_hopf, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
end

% Historical data
plot(19.3, 0.2, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'k');

xlabel('Wind Speed V (m/s)');
ylabel('Frequency (Hz)');
title('Flutter Frequency vs Wind Speed');
ylim([0, 0.5]);

legend_items = {'Flutter Frequency', 'Heaving f_h', 'Torsional f_theta'};
if ~isnan(V_hopf)
    legend_items = [legend_items, 'Hopf Point'];
end
legend_items = [legend_items, 'Historical'];
legend(legend_items, 'Location', 'best');
grid on;

% Subplot 3: Time series simulation
subplot(1,3,3);
t_span = [0, 60];

% Choose wind speeds relative to critical speed
if ~isnan(V_hopf)
    V_examples = [V_hopf - 5, V_hopf, V_hopf + 5];
    labels = {sprintf('V = %.1f m/s (stable)', V_examples(1)), ...
              sprintf('V = %.1f m/s (critical)', V_examples(2)), ...
              sprintf('V = %.1f m/s (unstable)', V_examples(3))};
else
    V_examples = [15, 20, 25];
    labels = {'V = 15 m/s', 'V = 20 m/s', 'V = 25 m/s'};
end

colors = ['b', 'k', 'r'];

for i = 1:length(V_examples)
    V = V_examples(i);
    
    % Build system matrix for this wind speed
    % Aerodynamic forces
    M_aero_hh = rho * B^2 * H1_star;
    M_aero_h_theta = rho * B^3 * H3_star;
    M_aero_theta_h = rho * B^3 * A1_star;
    M_aero_theta_theta = rho * B^4 * A3_star;
    
    C_aero_hh = rho * V * B^2 * H2_star;
    C_aero_h_theta = rho * V * B^3 * H4_star;
    C_aero_theta_h = rho * V * B^3 * A2_star;
    C_aero_theta_theta = rho * V * B^4 * A4_star;
    
    % Total system matrices
    M_total = [m + M_aero_hh,           M_aero_h_theta; ...
               M_aero_theta_h,          I + M_aero_theta_theta];
    
    C_total = [2*zeta_h*omega_h*m + C_aero_hh,     C_aero_h_theta; ...
               C_aero_theta_h,                     2*zeta_theta*omega_theta*I + C_aero_theta_theta];
    
    K_total = [m * omega_h^2,  0; ...
               0,              I * omega_theta^2];
    
    try
        M_inv = inv(M_total);
        A = [zeros(2,2),        eye(2); ...
             -M_inv * K_total,  -M_inv * C_total];
        
        % ODE system
        bridge_flutter = @(t, x) A * x;
        
        % Initial conditions (small disturbance)
        x0 = [0.001; 0; 0.001; 0];  % [h, h_dot, theta, theta_dot]
        
        % Check if system is stable
        eigenvals = eig(A);
        max_real = max(real(eigenvals));
        
        if max_real > 0.1  % Highly unstable
            % Shorter time span for unstable case
            [t, x] = ode45(bridge_flutter, [0, 30], x0, ...
                          odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
        else
            [t, x] = ode45(bridge_flutter, t_span, x0, ...
                          odeset('RelTol', 1e-6, 'AbsTol', 1e-8));
        end
        
        % Plot torsional motion in degrees
        plot(t, x(:,3)*180/pi, colors(i), 'LineWidth', 1.5); hold on;
        
    catch
        % If simulation fails, skip this case
        continue;
    end
end

xlabel('Time (s)');
ylabel('Torsional Angle theta (degrees)');
title('Bridge Response to Wind');
legend(labels, 'Location', 'best');
grid on;

% Add summary
fprintf('\n=== TACOMA BRIDGE FLUTTER ANALYSIS ===\n');
if ~isnan(V_hopf)
    fprintf('Predicted Hopf bifurcation at V = %.1f m/s\n', V_hopf);
    fprintf('Flutter frequency at bifurcation: %.2f Hz\n', freq_hopf);
    fprintf('Comparison to historical data:\n');
    fprintf('  Historical collapse wind speed: 19.3 m/s\n');
    fprintf('  Model prediction error: %.1f%%\n', abs(V_hopf - 19.3)/19.3 * 100);
else
    fprintf('No Hopf bifurcation found in analyzed wind speed range\n');
end
fprintf('Historical flutter frequency: ~0.2 Hz\n');
fprintf('\nThe Hopf bifurcation marks the transition from stable equilibrium\n');
fprintf('to self-sustaining oscillations (flutter). Above the critical wind\n');
fprintf('speed, small disturbances grow exponentially into large-amplitude\n');
fprintf('oscillations that can lead to structural failure.\n\n');

%% 2. POPULATION DYNAMICS: LOGISTIC GROWTH (Transcritical Bifurcation)
% dN/dt = rN(1 - N/K) - H, where H is harvesting rate
% Bifurcation parameter: r (growth rate)

figure('Name', 'Population Dynamics: Logistic Growth with Harvesting', 'Position', [150, 150, 1400, 500]);

% Parameters - better for visualization
K = 1000;   % Carrying capacity
H = 80;     % Constant harvesting rate
r_range = linspace(0.25, 0.45, 1000);  % Range adjusted around bifurcation

% Equilibrium analysis: rN(1 - N/K) - H = 0
% Rearranging: rN - rN²/K - H = 0 → rN²/K - rN + H = 0
% Solutions: N = (K/2) ± √((K/2)² - HK/r)
N_eq = zeros(length(r_range), 2);
for i = 1:length(r_range)
    r = r_range(i);
    discriminant = (K/2)^2 - H*K/r;
    if discriminant >= 0
        N_eq(i, 1) = (K/2) + sqrt(discriminant);  % Upper equilibrium
        N_eq(i, 2) = (K/2) - sqrt(discriminant);  % Lower equilibrium
    else
        N_eq(i, :) = NaN;  % No equilibrium (extinction)
    end
end

% Critical harvesting rate
r_critical = 4*H/K;

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
valid_idx = ~isnan(N_eq(:,1));

% Plot the curves properly
h1 = plot(r_range(valid_idx), N_eq(valid_idx,1), 'b-', 'LineWidth', 2); hold on;
h2 = plot(r_range(valid_idx), N_eq(valid_idx,2), 'r--', 'LineWidth', 2);
h3 = plot(r_critical, K/2, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

% Add extinction line for r < r_critical
h4 = plot([min(r_range), r_critical], [0, 0], 'g-', 'LineWidth', 2);

xlabel('Growth Rate r (1/year)');
ylabel('Population N');
title('Transcritical Bifurcation in Harvested Population');
legend([h1, h2, h3, h4], {'Stable Equilibrium', 'Unstable Equilibrium', 'Bifurcation Point', 'Extinction'}, 'Location', 'best');
grid on;

% Subplot 2: Phase portrait
subplot(1,3,2);
N_vals = linspace(0, 1200, 1000);
r_examples = [0.25, r_critical, 0.45];  % More separated values
colors = ['b', 'k', 'r'];

for i = 1:length(r_examples)
    r = r_examples(i);
    dN_dt = r*N_vals.*(1 - N_vals/K) - H;
    plot(N_vals, dN_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(N_vals, zeros(size(N_vals)), 'k--', 'LineWidth', 1);

xlabel('Population N');
ylabel('dN/dt (Growth Rate)');
title('Phase Portrait');
legend('r = 0.25', 'r = 0.32 (critical)', 'r = 0.45', 'Location', 'best');
grid on;

% Subplot 3: Time series
subplot(1,3,3);
t_span = [0, 50];
N0 = 800;  % Higher initial condition

% Store handles for proper legend
h = [];
for i = 1:length(r_examples)
    r = r_examples(i);
    
    % Simple event function to stop when population gets too low
    options_simple = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'Events', @(t,N) N - 1);
    
    try
        [t, N] = ode45(@(t,N) r*N*(1-N/K) - H, t_span, N0, options_simple);
        h(i) = plot(t, N, colors(i), 'LineWidth', 1.5); hold on;
    catch
        % If integration fails, try without events
        [t, N] = ode45(@(t,N) r*N*(1-N/K) - H, [0, 20], N0);
        h(i) = plot(t, N, colors(i), 'LineWidth', 1.5); hold on;
    end
end

xlabel('Time (years)');
ylabel('Population N');
title('Population Dynamics');
ylim([0, 1000]);
legend(h, {'r = 0.25', 'r = 0.32 (critical)', 'r = 0.45'}, 'Location', 'best');
grid on;

%% 3. NEUROSCIENCE: NEURON EXCITABILITY (Saddle-Node Bifurcation)
% Morris-Lecar model: dV/dt = (I - I_L - I_Ca - I_K)/C
% Bifurcation parameter: Applied current I

figure('Name', 'Neuroscience: Neuron Excitability', 'Position', [200, 200, 1400, 500]);

% Morris-Lecar parameters
C = 20;      % Membrane capacitance (μF/cm²)
g_L = 2;     % Leak conductance (mS/cm²)
g_Ca = 4.4;  % Calcium conductance (mS/cm²)
g_K = 8;     % Potassium conductance (mS/cm²)
V_L = -60;   % Leak reversal potential (mV)
V_Ca = 120;  % Calcium reversal potential (mV)
V_K = -84;   % Potassium reversal potential (mV)
V1 = -1.2;   % Half-activation voltage for Ca (mV)
V2 = 18;     % Slope parameter for Ca (mV)
V3 = 2;      % Half-activation voltage for K (mV)
V4 = 30;     % Slope parameter for K (mV)
phi = 0.04;  % Rate constant for K gating

% Applied current range
I_range = linspace(0, 200, 1000);

% Functions
m_inf = @(V) 0.5*(1 + tanh((V - V1)/V2));
n_inf = @(V) 0.5*(1 + tanh((V - V3)/V4));
tau_n = @(V) 1/(phi*cosh((V - V3)/(2*V4)));

% Find equilibrium points numerically
V_eq = zeros(size(I_range));
stability = zeros(size(I_range));

for i = 1:length(I_range)
    I = I_range(i);
    
    % Define system equations
    dV_dt = @(V, n) (I - g_L*(V - V_L) - g_Ca*m_inf(V)*(V - V_Ca) - g_K*n*(V - V_K))/C;
    dn_dt = @(V, n) (n_inf(V) - n)/tau_n(V);
    
    % Find equilibrium
    equilibrium = @(x) [dV_dt(x(1), x(2)); dn_dt(x(1), x(2))];
    try
        sol = fsolve(equilibrium, [-50, 0.1], optimset('Display', 'off'));
        V_eq(i) = sol(1);
        
        % Check stability via linearization
        h = 1e-6;
        J11 = (dV_dt(sol(1)+h, sol(2)) - dV_dt(sol(1)-h, sol(2)))/(2*h);
        J12 = (dV_dt(sol(1), sol(2)+h) - dV_dt(sol(1), sol(2)-h))/(2*h);
        J21 = (dn_dt(sol(1)+h, sol(2)) - dn_dt(sol(1)-h, sol(2)))/(2*h);
        J22 = (dn_dt(sol(1), sol(2)+h) - dn_dt(sol(1), sol(2)-h))/(2*h);
        
        eigenvals = eig([J11, J12; J21, J22]);
        if all(real(eigenvals) < 0)
            stability(i) = 1;  % Stable
        else
            stability(i) = -1; % Unstable
        end
    catch
        V_eq(i) = NaN;
        stability(i) = 0;
    end
end

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
stable_idx = stability == 1;
unstable_idx = stability == -1;

plot(I_range(stable_idx), V_eq(stable_idx), 'b-', 'LineWidth', 2); hold on;
plot(I_range(unstable_idx), V_eq(unstable_idx), 'r--', 'LineWidth', 2);

% Find approximate bifurcation point
bifurcation_idx = find(diff(stability) ~= 0, 1);
if ~isempty(bifurcation_idx)
    plot(I_range(bifurcation_idx), V_eq(bifurcation_idx), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
end

xlabel('Applied Current I (μA/cm²)');
ylabel('Membrane Potential V (mV)');
title('Neuron Excitability: Saddle-Node Bifurcation');
legend('Stable (Resting)', 'Unstable', 'Bifurcation Point', 'Location', 'best');
grid on;

% Subplot 2: Nullclines
subplot(1,3,2);
V_vals = linspace(-80, 40, 100);
I_examples = [40, 80, 120];

for i = 1:length(I_examples)
    I = I_examples(i);
    % V-nullcline: dV/dt = 0
    n_V_null = (I - g_L*(V_vals - V_L) - g_Ca*m_inf(V_vals).*(V_vals - V_Ca))./(g_K*(V_vals - V_K));
    plot(V_vals, n_V_null, colors(i), 'LineWidth', 2); hold on;
end

% n-nullcline: dn/dt = 0
n_n_null = n_inf(V_vals);
plot(V_vals, n_n_null, 'k--', 'LineWidth', 2);

xlabel('Membrane Potential V (mV)');
ylabel('K Channel Activation n');
title('Nullclines');
legend('I = 40', 'I = 80', 'I = 120', 'n-nullcline', 'Location', 'best');
grid on;

% Subplot 3: Action potential time series
subplot(1,3,3);
t_span = [0, 200];
I_examples = [30, 60, 90];  % Below, at, and above threshold

for i = 1:length(I_examples)
    I = I_examples(i);
    
    % System of ODEs
    morris_lecar = @(t, x) [...
        (I - g_L*(x(1) - V_L) - g_Ca*m_inf(x(1))*(x(1) - V_Ca) - g_K*x(2)*(x(1) - V_K))/C; ...
        (n_inf(x(1)) - x(2))/tau_n(x(1)) ...
    ];
    
    [t, sol] = ode45(morris_lecar, t_span, [-60, 0.1], options);
    plot(t, sol(:,1), colors(i), 'LineWidth', 1.5); hold on;
end

xlabel('Time (ms)');
ylabel('Membrane Potential V (mV)');
title('Action Potential Generation');
legend('I = 30 (subthreshold)', 'I = 60 (threshold)', 'I = 90 (suprathreshold)', 'Location', 'best');
grid on;

%% 4. CLIMATE SCIENCE: ARCTIC SEA ICE (Saddle-Node Bifurcation)
% Ice-albedo feedback model: dA/dt = -A + tanh(β(S - A))
% where A is ice area, S is solar forcing, β controls feedback strength

figure('Name', 'Climate Science: Arctic Sea Ice Tipping Point', 'Position', [250, 250, 1400, 500]);

% Parameters
beta = 5;     % Feedback strength
S_range = linspace(0, 2, 1000);

% Find equilibrium points: A = tanh(β(S - A))
A_eq = zeros(length(S_range), 3);  % Up to 3 equilibria possible

for i = 1:length(S_range)
    S = S_range(i);
    
    % Solve A = tanh(β(S - A)) numerically
    f = @(A) A - tanh(beta*(S - A));
    
    % Try different initial guesses
    try
        sol1 = fsolve(f, -0.9, optimset('Display', 'off'));
        sol2 = fsolve(f, 0, optimset('Display', 'off'));
        sol3 = fsolve(f, 0.9, optimset('Display', 'off'));
        
        solutions = [sol1, sol2, sol3];
        unique_sols = uniquetol(solutions, 1e-6);
        
        % Check stability: stable if |d/dA[A - tanh(β(S-A))]| < 1
        for j = 1:length(unique_sols)
            A = unique_sols(j);
            derivative = 1 + beta * (1 - tanh(beta*(S - A))^2);
            if abs(derivative) < 1
                A_eq(i, 1) = A;  % Stable branch
            else
                A_eq(i, 2) = A;  % Unstable branch
            end
        end
    catch
        A_eq(i, :) = NaN;
    end
end

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
plot(S_range, A_eq(:,1), 'b-', 'LineWidth', 2); hold on;
plot(S_range, A_eq(:,2), 'r--', 'LineWidth', 2);

% Find saddle-node bifurcation points
stable_points = ~isnan(A_eq(:,1));
transitions = find(diff(stable_points) ~= 0);

if ~isempty(transitions)
    for i = 1:length(transitions)
        idx = transitions(i);
        plot(S_range(idx), A_eq(idx,1), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    end
end

xlabel('Solar Forcing S');
ylabel('Ice Area A (normalized)');
title('Arctic Sea Ice: Saddle-Node Bifurcation');
legend('Stable Ice States', 'Unstable States', 'Tipping Points', 'Location', 'best');
grid on;

% Subplot 2: Hysteresis loop
subplot(1,3,2);
% Forward sweep (increasing S)
A_forward = zeros(size(S_range));
A_current = 0.8;  % Start with high ice coverage

for i = 1:length(S_range)
    S = S_range(i);
    f = @(A) A - tanh(beta*(S - A));
    
    try
        A_current = fsolve(f, A_current, optimset('Display', 'off'));
        A_forward(i) = A_current;
    catch
        A_forward(i) = NaN;
    end
end

% Backward sweep (decreasing S)
A_backward = zeros(size(S_range));
A_current = -0.8;  % Start with low ice coverage

for i = length(S_range):-1:1
    S = S_range(i);
    f = @(A) A - tanh(beta*(S - A));
    
    try
        A_current = fsolve(f, A_current, optimset('Display', 'off'));
        A_backward(i) = A_current;
    catch
        A_backward(i) = NaN;
    end
end

plot(S_range, A_forward, 'b-', 'LineWidth', 2); hold on;
plot(S_range, A_backward, 'r-', 'LineWidth', 2);

xlabel('Solar Forcing S');
ylabel('Ice Area A (normalized)');
title('Hysteresis in Ice Coverage');
legend('Forward (warming)', 'Backward (cooling)', 'Location', 'best');
grid on;

% Subplot 3: Time series under forcing
subplot(1,3,3);
t_span = [0, 100];
S_scenarios = [0.8, 1.2, 1.6];  % Different forcing levels

for i = 1:length(S_scenarios)
    S = S_scenarios(i);
    
    % ODE: dA/dt = -A + tanh(β(S - A))
    ice_dynamics = @(t, A) -A + tanh(beta*(S - A));
    
    [t, A] = ode45(ice_dynamics, t_span, 0.5, options);
    plot(t, A, colors(i), 'LineWidth', 1.5); hold on;
end

xlabel('Time (years)');
ylabel('Ice Area A (normalized)');
title('Ice Dynamics Under Different Forcings');
legend('S = 0.8 (stable ice)', 'S = 1.2 (bistable)', 'S = 1.6 (ice-free)', 'Location', 'best');
grid on;

%% 5. CHEMICAL REACTIONS: AUTOCATALYSIS (Transcritical Bifurcation)
% Autocatalytic reaction: A + X → 2X, X → P
% dx/dt = kax - bx², where a is substrate concentration

figure('Name', 'Chemical Reactions: Autocatalysis', 'Position', [300, 300, 1400, 500]);

% Parameters
k = 1;       % Rate constant
b = 0.1;     % Decay rate constant
a_range = linspace(0, 1, 1000);

% Equilibrium analysis: x* = 0 or x* = (ka - b)/b
x_eq1 = zeros(size(a_range));  % x* = 0
x_eq2 = k*a_range/b - 1;       % x* = (ka - b)/b

% Critical point: a_c = b/k
a_critical = b/k;

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
% x=0 branch stability
h1 = plot(a_range(a_range <= a_critical), zeros(size(a_range(a_range <= a_critical))), 'b-', 'LineWidth', 3); hold on;
h2 = plot(a_range(a_range >= a_critical), zeros(size(a_range(a_range >= a_critical))), 'r--', 'LineWidth', 3);

% x=(ka-b)/b branch for a >= a_critical
valid_idx = a_range >= a_critical;
plot(a_range(valid_idx), x_eq2(valid_idx), 'b-', 'LineWidth', 3);

h3 = plot(a_critical, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');

xlabel('Substrate Concentration a');
ylabel('Autocatalyst Concentration x');
title('Autocatalytic Reaction: Transcritical Bifurcation');
legend([h1, h2, h3], {'Stable', 'Unstable', 'Bifurcation Point'}, 'Location', 'best');
grid on;

% Subplot 2: Phase portrait
subplot(1,3,2);
x_vals = linspace(0, 3, 1000);
a_examples = [0.05, 0.1, 0.15];

for i = 1:length(a_examples)
    a = a_examples(i);
    dx_dt = k*a*x_vals - b*x_vals.^2;
    plot(x_vals, dx_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(x_vals, zeros(size(x_vals)), 'k--', 'LineWidth', 1);

xlabel('Autocatalyst Concentration x');
ylabel('dx/dt');
title('Phase Portrait');
legend('a = 0.05', 'a = 0.1 (critical)', 'a = 0.15', 'Location', 'best');
grid on;

% Subplot 3: Reaction kinetics
subplot(1,3,3);
t_span = [0, 20];
x0_vals = [0.1, 0.5, 1.0];

% Store plot handles for correct legend
h = [];
for i = 1:length(a_examples)
    a = a_examples(i);
    x0 = x0_vals(1);  % Use only one initial condition per parameter
    [t, x] = ode45(@(t,x) k*a*x - b*x^2, t_span, x0, options);
    h(i) = plot(t, x, colors(i), 'LineWidth', 1.5); hold on;
end

xlabel('Time');
ylabel('Autocatalyst Concentration x');
title('Reaction Kinetics');
legend(h, {'a = 0.05', 'a = 0.1 (critical)', 'a = 0.15'}, 'Location', 'best');
grid on;

%% Summary Display
fprintf('\n=== REAL-WORLD BIFURCATION APPLICATIONS ===\n');
fprintf('1. STRUCTURAL ENGINEERING: Euler buckling demonstrates supercritical pitchfork\n');
fprintf('   - Critical load: P_c = %.2f MN\n', P_critical/1e6);
fprintf('   - Above critical load: symmetric buckling modes become stable\n\n');

fprintf('2. POPULATION DYNAMICS: Logistic growth with harvesting shows transcritical bifurcation\n');
fprintf('   - Critical growth rate: r_c = %.4f /year\n', r_critical);
fprintf('   - Below critical rate: population extinction inevitable\n\n');

fprintf('3. NEUROSCIENCE: Morris-Lecar model exhibits saddle-node bifurcation\n');
fprintf('   - Threshold current determines excitability\n');
fprintf('   - Above threshold: action potential generation\n\n');

fprintf('4. CLIMATE SCIENCE: Arctic sea ice shows saddle-node bifurcation\n');
fprintf('   - Ice-albedo feedback creates tipping points\n');
fprintf('   - Hysteresis: different melting and freezing thresholds\n\n');

fprintf('5. CHEMICAL REACTIONS: Autocatalysis demonstrates transcritical bifurcation\n');
fprintf('   - Critical substrate concentration: a_c = %.2f\n', a_critical);
fprintf('   - Above threshold: autocatalytic growth dominates\n\n');

fprintf('These examples show how bifurcation theory predicts critical transitions\n');
fprintf('in diverse systems from engineering to biology to climate science.\n');

function [value, isterminal, direction] = eventfun(~, y)
    value = max(abs(y)) - 1e10;  % Stop if solution gets too large
    isterminal = 1;
    direction = 0;
end