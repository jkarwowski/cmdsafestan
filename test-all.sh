#!/usr/bin/env bash

if [ "$#" -ne 1 ]; then
RUNTESTARGS=""
else
RUNTESTARGS="-j"$1
fi

echo 'Running:'
echo '  - CmdStan tests'
echo '  - SafeStan tests'
echo '  - SafeStan Math Library tests'
echo ''
echo '------------------------------------------------------------'
echo 'CmdStan tests'
./runCmdStanTests.py $RUNTESTARGS src/test/interface


echo ''
echo '------------------------------------------------------------'
echo 'SafeStan tests'
pushd safestan/
./runTests.py $RUNTESTARGS src/test
popd


echo ''
echo '------------------------------------------------------------'
echo 'SafeStan Math Library tests'
pushd safestan/lib/stan_math/
./runTests.py $RUNTESTARGS test/unit
./runTests.py $RUNTESTARGS test/prob
popd
