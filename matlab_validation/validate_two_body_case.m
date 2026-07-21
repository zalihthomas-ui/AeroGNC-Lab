function result = validate_two_body_case()
%VALIDATE_TWO_BODY_CASE Independently integrate the shared two-body case.
% Uses MATLAB ode113, not the Python universal-variable implementation.

root = fileparts(mfilename('fullpath'));
inputPath = fullfile(root, 'two_body_case.json');
outputPath = fullfile(root, 'two_body_matlab_result.json');
data = jsondecode(fileread(inputPath));

initialState = [data.initial_position_m(:); data.initial_velocity_mps(:)];
options = odeset('RelTol', 1.0e-12, 'AbsTol', 1.0e-11, 'MaxStep', 10.0);
[time_s, state] = ode113( ...
    @(timeValue, stateValue) twoBodyDerivative(timeValue, stateValue, ...
    data.gravitational_parameter_m3_s2), ...
    [0.0, data.duration_s], initialState, options);
finalState = state(end, :).';

positionError_m = norm(finalState(1:3) - data.python_position_m(:));
velocityError_mps = norm(finalState(4:6) - data.python_velocity_mps(:));
passed = positionError_m <= data.absolute_position_tolerance_m && ...
    velocityError_mps <= data.absolute_velocity_tolerance_mps;

result = struct( ...
    'solver', 'MATLAB ode113 R2024a', ...
    'sample_count', numel(time_s), ...
    'final_position_m', finalState(1:3).', ...
    'final_velocity_mps', finalState(4:6).', ...
    'position_error_m', positionError_m, ...
    'velocity_error_mps', velocityError_mps, ...
    'position_tolerance_m', data.absolute_position_tolerance_m, ...
    'velocity_tolerance_mps', data.absolute_velocity_tolerance_mps, ...
    'passed', passed);

fileIdentifier = fopen(outputPath, 'w');
cleanup = onCleanup(@() fclose(fileIdentifier));
fwrite(fileIdentifier, jsonencode(result, PrettyPrint=true), 'char');
fprintf('TWO_BODY_POSITION_ERROR_M=%.12g\n', positionError_m);
fprintf('TWO_BODY_VELOCITY_ERROR_MPS=%.12g\n', velocityError_mps);
fprintf('TWO_BODY_PASS=%d\n', passed);
assert(passed, 'AeroGNC:TwoBodyValidationFailed', ...
    'MATLAB two-body result exceeded the documented tolerance.');
end

function derivative = twoBodyDerivative(~, state, gravitationalParameter_m3_s2)
position_m = state(1:3);
radius_m = norm(position_m);
acceleration_mps2 = -gravitationalParameter_m3_s2 * position_m / radius_m^3;
derivative = [state(4:6); acceleration_mps2];
end
