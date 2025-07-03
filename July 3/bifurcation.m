% Bifurcation Analysis: Demonstrations of Common Bifurcations
% This script demonstrates saddle-node, transcritical, pitchfork, and Hopf bifurcations
% ALL SOLUTIONS ARE PURELY ANALYTICAL - NO NUMERICAL INTEGRATION

clear; clc; close all;

%% 1. SADDLE-NODE BIFURCATION
% Normal form: dx/dt = r + x^2
figure('Name', 'Saddle-Node Bifurcation', 'Position', [100, 100, 1200, 400]);

% Parameter range
r_vals = linspace(-1, 1, 1000);
x_vals = linspace(-2, 2, 1000);

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
r_stable = r_vals(r_vals <= 0);

% Equilibrium points
x_eq_stable = -sqrt(-r_stable);
x_eq_unstable = sqrt(-r_stable);

plot(r_stable, x_eq_stable, 'b-', 'LineWidth', 2); hold on;
plot(r_stable, x_eq_unstable, 'r--', 'LineWidth', 2);
plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('x*');
xlim([-0.6, 0.4]);
title('Saddle-Node Bifurcation Diagram');
legend('Stable', 'Unstable', 'Bifurcation Point', 'Location', 'best');
grid on;

% Subplot 2: Vector field for different r values
subplot(1,3,2);
r_examples = [-0.5, 0, 0.5];
colors = ['b', 'k', 'r'];
for i = 1:length(r_examples)
    r = r_examples(i);
    dx_dt = r + x_vals.^2;
    plot(x_vals, dx_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(x_vals, zeros(size(x_vals)), 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('dx/dt');
title('Vector Fields');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;

% Subplot 3: Time series - completely analytical, no singularities
subplot(1,3,3);
t_vals = linspace(0, 4, 300);
colors = ['b', 'k', 'r'];

% r = -0.5: Stable case
t_stable = t_vals;
x_stable = -sqrt(0.5) * ones(size(t_stable)) + 0.3 * exp(-sqrt(0.5) * t_stable);
plot(t_stable, x_stable, colors(1), 'LineWidth', 2); hold on;

% r = 0: Critical case  
t_critical = t_vals;
x_critical = -1 ./ (1 + 0.5 * t_critical);
plot(t_critical, x_critical, colors(2), 'LineWidth', 2);

% r = 0.5: Unstable case - limited time
t_unstable = linspace(0, 1, 100);
x_unstable = 0.1 * tan(sqrt(0.5) * t_unstable);
plot(t_unstable, x_unstable, colors(3), 'LineWidth', 2);

xlabel('Time t'); ylabel('x(t)');
title('Time Series Solutions');
legend('r = -0.5 (stable)', 'r = 0 (critical)', 'r = 0.5 (unstable)', 'Location', 'best');
grid on;
ylim([-1.5, 1.5]);

%% 2. TRANSCRITICAL BIFURCATION
% Normal form: dx/dt = rx - x^2
figure('Name', 'Transcritical Bifurcation', 'Position', [150, 150, 1200, 400]);

% Subplot 1: Bifurcation diagram
subplot(1,3,1);
r_vals = linspace(-1, 1, 1000);

% Stability analysis
r_neg = r_vals(r_vals <= 0);
r_pos = r_vals(r_vals >= 0);
plot(r_neg, zeros(size(r_neg)), 'b-', 'LineWidth', 3); hold on;
plot(r_pos, zeros(size(r_pos)), 'r--', 'LineWidth', 3);
plot(r_neg, r_neg, 'r--', 'LineWidth', 3);
plot(r_pos, r_pos, 'b-', 'LineWidth', 3);

plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('x*');
title('Transcritical Bifurcation Diagram');
legend('Stable', 'Unstable', 'Location', 'best');
grid on;

% Subplot 2: Vector field
subplot(1,3,2);
x_vals = linspace(-2, 2, 1000);
r_examples = [-0.5, 0, 0.5];
for i = 1:length(r_examples)
    r = r_examples(i);
    dx_dt = r*x_vals - x_vals.^2;
    plot(x_vals, dx_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(x_vals, zeros(size(x_vals)), 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('dx/dt');
title('Vector Fields');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;

% Subplot 3: Time series - safe analytical solutions
subplot(1,3,3);
t_vals = linspace(0, 6, 300);

% r = -0.5: x=0 stable
x_r1 = 0.4 * exp(-0.5 * t_vals);
plot(t_vals, x_r1, colors(1), 'LineWidth', 2); hold on;

% r = 0: Critical case
x_r2 = 0.3 ./ (1 + 0.3 * t_vals);
plot(t_vals, x_r2, colors(2), 'LineWidth', 2);

% r = 0.5: Approach to x=r
x_r3 = 0.5 * (1 - exp(-0.5 * t_vals));
plot(t_vals, x_r3, colors(3), 'LineWidth', 2);

xlabel('Time t'); ylabel('x(t)');
title('Time Series Solutions');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;
ylim([-0.1, 0.6]);

%% 3. PITCHFORK BIFURCATIONS
figure('Name', 'Pitchfork Bifurcations', 'Position', [200, 200, 1200, 800]);

% Supercritical Pitchfork: dx/dt = rx - x^3
subplot(2,3,1);
r_vals = linspace(-1, 1, 1000);
x_eq1 = zeros(size(r_vals));  % x* = 0
x_eq2 = sqrt(max(0, r_vals));  % x* = +sqrt(r) for r > 0
x_eq3 = -sqrt(max(0, r_vals)); % x* = -sqrt(r) for r > 0

% Plot stable and unstable branches
h1 = plot(r_vals(r_vals <= 0), zeros(size(r_vals(r_vals <= 0))), 'b-', 'LineWidth', 3); hold on;
h2 = plot(r_vals(r_vals >= 0), zeros(size(r_vals(r_vals >= 0))), 'r--', 'LineWidth', 3);

r_pos = r_vals(r_vals >= 0);
plot(r_pos, sqrt(r_pos), 'b-', 'LineWidth', 3);
plot(r_pos, -sqrt(r_pos), 'b-', 'LineWidth', 3);
h3 = plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('x*');
title('Supercritical Pitchfork');
legend([h1, h2, h3], {'Stable', 'Unstable', 'Bifurcation Point'}, 'Location', 'best');
grid on;

% Subcritical Pitchfork: dx/dt = rx + x^3
subplot(2,3,4);
% r < 0: x=0 stable (blue), x=±√(-r) unstable (red dashed)  
% r > 0: x=0 unstable (red dashed)

h1 = plot(r_vals(r_vals < 0), zeros(size(r_vals(r_vals < 0))), 'b-', 'LineWidth', 3); hold on;
h2 = plot(r_vals(r_vals > 0), zeros(size(r_vals(r_vals > 0))), 'r--', 'LineWidth', 3);

% x=±√(-r) branches for r < 0 are UNSTABLE (red dashed)
r_neg = r_vals(r_vals < 0);
plot(r_neg, sqrt(-r_neg), 'r--', 'LineWidth', 3);
plot(r_neg, -sqrt(-r_neg), 'r--', 'LineWidth', 3);

h3 = plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('x*');
title('Subcritical Pitchfork');
legend([h1, h2, h3], {'Stable', 'Unstable', 'Bifurcation Point'}, 'Location', 'best');
grid on;

% Vector fields for supercritical
subplot(2,3,2);
x_vals = linspace(-2, 2, 1000);
r_examples = [-0.5, 0, 0.5];
for i = 1:length(r_examples)
    r = r_examples(i);
    dx_dt = r*x_vals - x_vals.^3;
    plot(x_vals, dx_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(x_vals, zeros(size(x_vals)), 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('dx/dt');
title('Supercritical Vector Fields');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;

% Vector fields for subcritical
subplot(2,3,5);
for i = 1:length(r_examples)
    r = r_examples(i);
    dx_dt = r*x_vals + x_vals.^3;
    plot(x_vals, dx_dt, colors(i), 'LineWidth', 2); hold on;
end
plot(x_vals, zeros(size(x_vals)), 'k--', 'LineWidth', 1);
xlabel('x'); ylabel('dx/dt');
title('Subcritical Vector Fields');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;

% Time series for supercritical
subplot(2,3,3);
t_vals = linspace(0, 5, 300);

% r = -0.5: Decay to x=0
x_sup1 = 0.5 * exp(-0.5 * t_vals);
plot(t_vals, x_sup1, colors(1), 'LineWidth', 2); hold on;

% r = 0: Critical decay
x_sup2 = 0.4 ./ sqrt(1 + 0.32 * t_vals);
plot(t_vals, x_sup2, colors(2), 'LineWidth', 2);

% r = 0.5: Approach to equilibrium
x_sup3 = sqrt(0.5) * tanh(sqrt(0.5) * t_vals);
plot(t_vals, x_sup3, colors(3), 'LineWidth', 2);

xlabel('Time t'); ylabel('x(t)');
title('Supercritical Time Series');
legend('r = -0.5', 'r = 0', 'r = 0.5', 'Location', 'best');
grid on;
ylim([-0.1, 0.8]);

% Time series for subcritical
subplot(2,3,6);
% Only show stable cases
x_sub1 = 0.3 * exp(-0.5 * t_vals);
plot(t_vals, x_sub1, colors(1), 'LineWidth', 2); hold on;

x_sub2 = 0.2 ./ sqrt(1 + 0.08 * t_vals);
plot(t_vals, x_sub2, colors(2), 'LineWidth', 2);

xlabel('Time t'); ylabel('x(t)');
title('Subcritical Time Series');
legend('r = -0.5', 'r = 0', 'Location', 'best');
grid on;
ylim([-0.05, 0.35]);

%% 4. HOPF BIFURCATIONS - COMPLETELY ANALYTICAL
figure('Name', 'Hopf Bifurcations', 'Position', [250, 250, 1400, 800]);

% Supercritical Hopf bifurcation diagram
subplot(2,4,1);
r_vals = linspace(-1, 1, 1000);
plot(r_vals, zeros(size(r_vals)), 'k-', 'LineWidth', 2); hold on;
plot(r_vals(r_vals <= 0), zeros(size(r_vals(r_vals <= 0))), 'b-', 'LineWidth', 3);
plot(r_vals(r_vals >= 0), zeros(size(r_vals(r_vals >= 0))), 'r--', 'LineWidth', 3);

r_pos = r_vals(r_vals > 0);
plot(r_pos, sqrt(r_pos), 'b-', 'LineWidth', 3);
plot(r_pos, -sqrt(r_pos), 'b-', 'LineWidth', 3);
plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('Amplitude');
title('Supercritical Hopf');
legend('Stable Fixed Point', 'Unstable Fixed Point', 'Stable Limit Cycle', 'Location', 'best');
grid on; ylim([-1.5, 1.5]);

% Subcritical Hopf bifurcation diagram
subplot(2,4,5);
plot(r_vals, zeros(size(r_vals)), 'k-', 'LineWidth', 2); hold on;
plot(r_vals(r_vals >= 0), zeros(size(r_vals(r_vals >= 0))), 'r--', 'LineWidth', 3);
plot(r_vals(r_vals <= 0), zeros(size(r_vals(r_vals <= 0))), 'b-', 'LineWidth', 3);

r_neg = r_vals(r_vals < 0);
plot(r_neg, sqrt(-r_neg), 'r--', 'LineWidth', 3);
plot(r_neg, -sqrt(-r_neg), 'r--', 'LineWidth', 3);
plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('Parameter r'); ylabel('Amplitude');
title('Subcritical Hopf');
legend('Stable Fixed Point', 'Unstable Fixed Point', 'Unstable Limit Cycle', 'Location', 'best');
grid on; ylim([-1.5, 1.5]);

% Phase portraits - purely analytical
r_examples = [-0.5, 0, 0.5];
titles = {'r = -0.5 (Stable Focus)', 'r = 0 (Center)', 'r = 0.5 (Limit Cycle)'};

for i = 1:3
    subplot(2,4,i+1);
    r = r_examples(i);
    
    % Vector field
    [x_grid, y_grid] = meshgrid(-2.5:0.4:2.5, -2.5:0.4:2.5);
    dx = r*x_grid - y_grid - x_grid.*(x_grid.^2 + y_grid.^2);
    dy = x_grid + r*y_grid - y_grid.*(x_grid.^2 + y_grid.^2);
    
    magnitude = sqrt(dx.^2 + dy.^2);
    dx_norm = dx./(magnitude + 1e-10);
    dy_norm = dy./(magnitude + 1e-10);
    quiver(x_grid, y_grid, dx_norm, dy_norm, 0.4, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
    hold on;
    
    % Analytical trajectories
    if r <= 0
        % Stable spirals
        radii = [0.3, 0.6, 0.9];
        for radius = radii
            t_spiral = linspace(0, 15, 200);
            rho = radius * exp(r * t_spiral);
            theta = t_spiral;
            x_traj = rho .* cos(theta);
            y_traj = rho .* sin(theta);
            valid_idx = (abs(x_traj) < 2.5) & (abs(y_traj) < 2.5);
            plot(x_traj(valid_idx), y_traj(valid_idx), 'b-', 'LineWidth', 1.5);
        end
    elseif r > 0
        % Limit cycle
        theta_lc = linspace(0, 2*pi, 100);
        x_lc = sqrt(r)*cos(theta_lc);
        y_lc = sqrt(r)*sin(theta_lc);
        plot(x_lc, y_lc, 'r-', 'LineWidth', 3);
        
        % Spirals approaching limit cycle
        radii = [0.3, 1.2];
        for radius = radii
            t_spiral = linspace(0, 10, 200);
            rho = sqrt(r) + (radius - sqrt(r)) * exp(-r * t_spiral);
            theta = t_spiral;
            x_traj = rho .* cos(theta);
            y_traj = rho .* sin(theta);
            plot(x_traj, y_traj, 'b-', 'LineWidth', 1.5);
        end
    end
    
    plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
    xlabel('x'); ylabel('y');
    title(titles{i});
    axis equal; grid on;
    xlim([-2.5, 2.5]); ylim([-2.5, 2.5]);
end

% Subcritical Hopf phase portraits
subplot(2,4,6);
r = -0.5;
[x_grid, y_grid] = meshgrid(-2.5:0.4:2.5, -2.5:0.4:2.5);
dx = r*x_grid - y_grid + x_grid.*(x_grid.^2 + y_grid.^2);
dy = x_grid + r*y_grid + y_grid.*(x_grid.^2 + y_grid.^2);
magnitude = sqrt(dx.^2 + dy.^2);
dx_norm = dx./(magnitude + 1e-10);
dy_norm = dy./(magnitude + 1e-10);
quiver(x_grid, y_grid, dx_norm, dy_norm, 0.4, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
hold on;

% Unstable limit cycle
theta_lc = linspace(0, 2*pi, 100);
x_lc = sqrt(-r)*cos(theta_lc);
y_lc = sqrt(-r)*sin(theta_lc);
plot(x_lc, y_lc, 'r--', 'LineWidth', 3);

% Stable spirals
radii = [0.3, 0.6];
for radius = radii
    t_spiral = linspace(0, 10, 200);
    rho = radius * exp(r * t_spiral);
    theta = t_spiral;
    x_traj = rho .* cos(theta);
    y_traj = rho .* sin(theta);
    valid_idx = (abs(x_traj) < 2.5) & (abs(y_traj) < 2.5);
    plot(x_traj(valid_idx), y_traj(valid_idx), 'b-', 'LineWidth', 1.5);
end

plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('x'); ylabel('y');
title('Subcritical: r = -0.5');
axis equal; grid on;
xlim([-2.5, 2.5]); ylim([-2.5, 2.5]);

subplot(2,4,7);
r = 0;
[x_grid, y_grid] = meshgrid(-2.5:0.4:2.5, -2.5:0.4:2.5);
dx = -y_grid;
dy = x_grid;
magnitude = sqrt(dx.^2 + dy.^2);
dx_norm = dx./(magnitude + 1e-10);
dy_norm = dy./(magnitude + 1e-10);
quiver(x_grid, y_grid, dx_norm, dy_norm, 0.4, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
hold on;

% Circular trajectories
radii = [0.3, 0.6, 0.9];
for radius = radii
    theta_circle = linspace(0, 2*pi, 100);
    x_circle = radius*cos(theta_circle);
    y_circle = radius*sin(theta_circle);
    plot(x_circle, y_circle, 'b-', 'LineWidth', 1.5);
end

plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('x'); ylabel('y');
title('Subcritical: r = 0');
axis equal; grid on;
xlim([-2.5, 2.5]); ylim([-2.5, 2.5]);

subplot(2,4,8);
r = 0.5;
[x_grid, y_grid] = meshgrid(-2.5:0.4:2.5, -2.5:0.4:2.5);
dx = r*x_grid - y_grid + x_grid.*(x_grid.^2 + y_grid.^2);
dy = x_grid + r*y_grid + y_grid.*(x_grid.^2 + y_grid.^2);
magnitude = sqrt(dx.^2 + dy.^2);
dx_norm = dx./(magnitude + 1e-10);
dy_norm = dy./(magnitude + 1e-10);
quiver(x_grid, y_grid, dx_norm, dy_norm, 0.4, 'Color', [0.7, 0.7, 0.7], 'LineWidth', 0.5);
hold on;

% Diverging trajectories
radii = [0.1, 0.05];
for radius = radii
    t_spiral = linspace(0, 2, 100);
    rho = radius * exp(r * t_spiral);
    theta = t_spiral;
    x_traj = rho .* cos(theta);
    y_traj = rho .* sin(theta);
    valid_idx = (abs(x_traj) < 2.5) & (abs(y_traj) < 2.5);
    plot(x_traj(valid_idx), y_traj(valid_idx), 'b-', 'LineWidth', 1.5);
end

plot(0, 0, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
xlabel('x'); ylabel('y');
title('Subcritical: r = 0.5');
axis equal; grid on;
xlim([-2.5, 2.5]); ylim([-2.5, 2.5]);

%% Summary Display
fprintf('\n=== BIFURCATION ANALYSIS SUMMARY ===\n');
fprintf('1. SADDLE-NODE: Two equilibria collide and annihilate\n');
fprintf('   Normal form: dx/dt = r + x^2\n');
fprintf('   Bifurcation at r = 0\n\n');

fprintf('2. TRANSCRITICAL: Equilibria exchange stability\n');
fprintf('   Normal form: dx/dt = rx - x^2\n');
fprintf('   Bifurcation at r = 0\n\n');

fprintf('3. PITCHFORK: Symmetry-breaking bifurcation\n');
fprintf('   Supercritical: dx/dt = rx - x^3 (stable branches)\n');
fprintf('   Subcritical: dx/dt = rx + x^3 (unstable branches)\n');
fprintf('   Bifurcation at r = 0\n\n');

fprintf('4. HOPF: Birth/death of limit cycles\n');
fprintf('   Supercritical: Stable limit cycle emerges\n');
fprintf('   Subcritical: Unstable limit cycle disappears\n');
fprintf('   Bifurcation at r = 0\n\n');

fprintf('All solutions are purely analytical - no numerical integration used.\n');
fprintf('Blue = stable, Red dashed = unstable, Black dot = bifurcation point\n');