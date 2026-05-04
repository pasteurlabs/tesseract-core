function spring_mass_damper(input_file, output_file)
%SPRING_MASS_DAMPER Solve a spring-mass-damper ODE system.
%
%   Solves the damped harmonic oscillator equation:
%       m * x'' + c * x' + k * x = F0  (step force input)
%
%   using MATLAB's ode45 (Dormand-Prince Runge-Kutta).
%
%   Arguments:
%       input_file  - Path to JSON file with input parameters
%       output_file - Path to write JSON results

    % Read input parameters
    raw = fileread(input_file);
    params = jsondecode(raw);

    mass = params.mass;
    damping = params.damping;
    stiffness = params.stiffness;
    force_amplitude = params.force_amplitude;
    t_end = params.t_end;
    n_output_points = params.n_output_points;

    % Define the ODE system as first-order:
    %   y(1) = x   (displacement)
    %   y(2) = x'  (velocity)
    %
    %   y(1)' = y(2)
    %   y(2)' = (F0 - c*y(2) - k*y(1)) / m
    odefun = @(t, y) [
        y(2);
        (force_amplitude - damping * y(2) - stiffness * y(1)) / mass
    ];

    % Initial conditions: at rest
    y0 = [0; 0];

    % Output time points
    t_eval = linspace(0, t_end, n_output_points);

    % Solve with ode45
    opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
    [t_sol, y_sol] = ode45(odefun, t_eval, y0, opts);

    % Build output structure
    result = struct();
    result.time = t_sol(:)';
    result.displacement = y_sol(:, 1)';
    result.velocity = y_sol(:, 2)';

    % Write output
    json_str = jsonencode(result);
    fid = fopen(output_file, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);

    fprintf('Solver completed: %d time steps, t_end = %.3f s\n', ...
        n_output_points, t_end);
end
