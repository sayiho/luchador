#!/bin/sh
set -u

if [ ${COUNT_INTEGRATION_COVERAGE:-false} = true ]; then
    TEST_COMMAND="coverage run --parallel-mode"
else
    TEST_COMMAND="python"
fi
PORT="5010"
PID_FILE=tmp/parameter_server_pid.txt

echo "Launching Parameter Server"
python tests/integration/test_server_client/launch_parameter_server.py --port ${PORT} &
PID="$!"
sleep 3;
echo "Testing client"
python tests/integration/test_server_client/test_parameter_client.py --port ${PORT}
echo "Killing Server"
kill "${PID}"

