function modelPath = build_attitude_channel_model()
%BUILD_ATTITUDE_CHANNEL_MODEL Build a small independent one-axis validation model.
%   The generated model contains a cascaded proportional attitude/rate controller
%   and a damped rigid-body attitude channel. It requires Simulink but no specialised
%   control toolbox. No result should be claimed unless run_attitude_channel_validation
%   has executed successfully.

if exist('sim', 'file') ~= 2
    error('AeroGNC:SimulinkUnavailable', ...
          'Simulink is not installed in this MATLAB environment.');
end

scriptDirectory = fileparts(mfilename('fullpath'));
modelName = 'aerognc_attitude_channel';
modelPath = fullfile(scriptDirectory, [modelName '.slx']);
if bdIsLoaded(modelName)
    close_system(modelName, 0);
end
new_system(modelName);

add_block('simulink/Sources/Step', [modelName '/Attitude reference'], ...
          'Time', '0.25', 'Before', '0', 'After', num2str(5*pi/180, 17), ...
          'Position', [35 65 65 95]);
add_block('simulink/Math Operations/Sum', [modelName '/Attitude error'], ...
          'Inputs', '+-', 'Position', [105 61 125 99]);
add_block('simulink/Math Operations/Gain', [modelName '/Attitude gain'], ...
          'Gain', '4.0', 'Position', [165 65 225 95]);
add_block('simulink/Math Operations/Sum', [modelName '/Rate error'], ...
          'Inputs', '+-', 'Position', [265 61 285 99]);
add_block('simulink/Math Operations/Gain', [modelName '/Rate gain'], ...
          'Gain', '12.0', 'Position', [325 65 385 95]);
add_block('simulink/Continuous/State-Space', [modelName '/Rigid body plant'], ...
          'A', '[0 1; 0 -0.8/2.4]', 'B', '[0; 1/2.4]', ...
          'C', 'eye(2)', 'D', '[0; 0]', 'X0', '[0; 0]', ...
          'Position', [435 57 525 103]);
add_block('simulink/Signal Routing/Demux', [modelName '/State demux'], ...
          'Outputs', '2', 'Position', [570 52 575 108]);
add_block('simulink/Sinks/To Workspace', [modelName '/theta log'], ...
          'VariableName', 'theta_log', 'SaveFormat', 'Timeseries', ...
          'Position', [650 35 745 65]);
add_block('simulink/Sinks/To Workspace', [modelName '/rate log'], ...
          'VariableName', 'rate_log', 'SaveFormat', 'Timeseries', ...
          'Position', [650 95 745 125]);
add_block('simulink/Sinks/To Workspace', [modelName '/command log'], ...
          'VariableName', 'command_log', 'SaveFormat', 'Timeseries', ...
          'Position', [435 130 530 160]);

add_line(modelName, 'Attitude reference/1', 'Attitude error/1');
add_line(modelName, 'Attitude error/1', 'Attitude gain/1');
add_line(modelName, 'Attitude gain/1', 'Rate error/1');
add_line(modelName, 'Rate error/1', 'Rate gain/1');
add_line(modelName, 'Rate gain/1', 'Rigid body plant/1');
add_line(modelName, 'Rate gain/1', 'command log/1', 'autorouting', 'on');
add_line(modelName, 'Rigid body plant/1', 'State demux/1');
add_line(modelName, 'State demux/1', 'theta log/1');
add_line(modelName, 'State demux/1', 'Attitude error/2', 'autorouting', 'on');
add_line(modelName, 'State demux/2', 'rate log/1');
add_line(modelName, 'State demux/2', 'Rate error/2', 'autorouting', 'on');

set_param(modelName, 'SolverType', 'Fixed-step', 'Solver', 'ode4', ...
          'FixedStep', '0.001', 'StopTime', '5.0');
save_system(modelName, modelPath);
close_system(modelName);
fprintf('Built %s\n', modelPath);
end
