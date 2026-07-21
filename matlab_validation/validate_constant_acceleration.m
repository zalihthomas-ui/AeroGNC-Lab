function report = validate_constant_acceleration()
%VALIDATE_CONSTANT_ACCELERATION Independent fixed-step RK4 translation check.
%   Reads the same JSON case used by Python, integrates a six-state NED
%   constant-force problem, checks the analytical solution, and writes CSV/JSON
%   evidence under matlab_validation/output. MATLAB is never required by the
%   Python package.

scriptDirectory = fileparts(mfilename('fullpath'));
casePath = fullfile(scriptDirectory, 'constant_acceleration_case.json');
configuration = jsondecode(fileread(casePath));

duration_s = configuration.duration_s;
step_s = configuration.time_step_s;
stepCount = round(duration_s / step_s);
time_s = (0:stepCount)' * step_s;

state = zeros(stepCount + 1, 6);
state(1, :) = [configuration.initial_position_ned_m(:); ...
               configuration.initial_velocity_ned_mps(:)]';
acceleration_ned_mps2 = configuration.force_ned_N(:) / configuration.mass_kg ...
                        + configuration.gravity_ned_mps2(:);
derivative = @(value) [value(4:6); acceleration_ned_mps2];

for index = 1:stepCount
    current = state(index, :)';
    k1 = derivative(current);
    k2 = derivative(current + 0.5 * step_s * k1);
    k3 = derivative(current + 0.5 * step_s * k2);
    k4 = derivative(current + step_s * k3);
    state(index + 1, :) = (current + step_s * (k1 + 2*k2 + 2*k3 + k4) / 6)';
end

initialPosition = configuration.initial_position_ned_m(:)';
initialVelocity = configuration.initial_velocity_ned_mps(:)';
exactPosition = initialPosition + time_s .* initialVelocity ...
                + 0.5 * time_s.^2 .* acceleration_ned_mps2';
exactVelocity = initialVelocity + time_s .* acceleration_ned_mps2';
exactState = [exactPosition, exactVelocity];
maxAnalyticalError = max(abs(state - exactState), [], 'all');

outputDirectory = fullfile(scriptDirectory, 'output');
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end
tableOutput = array2table([time_s, state], 'VariableNames', { ...
    'time_s', 'north_m', 'east_m', 'down_m', 'north_velocity_mps', ...
    'east_velocity_mps', 'down_velocity_mps'});
writetable(tableOutput, fullfile(outputDirectory, 'constant_acceleration_matlab.csv'));

report = struct( ...
    'matlab_version', version, ...
    'case_name', configuration.case_name, ...
    'sample_count', height(tableOutput), ...
    'matlab_analytic_max_abs_error', maxAnalyticalError, ...
    'tolerance', configuration.absolute_tolerance, ...
    'passed', maxAnalyticalError <= configuration.absolute_tolerance);
reportPath = fullfile(outputDirectory, 'matlab_execution_report.json');
fileIdentifier = fopen(reportPath, 'w');
cleaner = onCleanup(@() fclose(fileIdentifier));
fprintf(fileIdentifier, '%s\n', jsonencode(report, PrettyPrint=true));

fprintf('MATLAB analytical maximum error: %.3e (tolerance %.3e)\n', ...
        maxAnalyticalError, configuration.absolute_tolerance);
if ~report.passed
    error('AeroGNC:ValidationFailure', 'MATLAB constant-acceleration check failed.');
end
end
