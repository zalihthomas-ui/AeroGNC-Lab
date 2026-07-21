function report = run_attitude_channel_validation()
%RUN_ATTITUDE_CHANNEL_VALIDATION Execute and export the generated Simulink case.

scriptDirectory = fileparts(mfilename('fullpath'));
modelPath = build_attitude_channel_model();
[~, modelName] = fileparts(modelPath);
load_system(modelPath);
simulationOutput = sim(modelName, 'ReturnWorkspaceOutputs', 'on');

theta = simulationOutput.theta_log;
rate = simulationOutput.rate_log;
command = simulationOutput.command_log;
outputDirectory = fullfile(scriptDirectory, 'output');
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end
sampleTime = theta.Time;
tableOutput = table(sampleTime, theta.Data, rate.Data, command.Data, ...
                    'VariableNames', {'time_s', 'attitude_rad', ...
                                      'angular_rate_radps', 'command_Nm'});
writetable(tableOutput, fullfile(outputDirectory, 'attitude_channel_simulink.csv'));

reference_rad = 5*pi/180;
steadyStateError = abs(theta.Data(end) - reference_rad);
report = struct( ...
    'matlab_version', version, ...
    'model', modelName, ...
    'sample_count', height(tableOutput), ...
    'steady_state_error_rad', steadyStateError, ...
    'executed', true);
fileIdentifier = fopen(fullfile(outputDirectory, 'simulink_execution_report.json'), 'w');
cleaner = onCleanup(@() fclose(fileIdentifier));
fprintf(fileIdentifier, '%s\n', jsonencode(report, PrettyPrint=true));
close_system(modelName, 0);
fprintf('Simulink final attitude error: %.3e rad\n', steadyStateError);
end
