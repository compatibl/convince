@echo off

pushd ..

echo.
echo Delete build, egg-info, and dist directories before building
if exist convince.egg-info rmdir/q/s convince.egg-info
if exist build rmdir/q/s build
if exist dist rmdir/q/s dist

echo.
echo Build wheel
python setup.py bdist_wheel

echo.
echo Delete build and egg-info directories after building, wheel is located in dist
if exist convince.egg-info rmdir/q/s convince.egg-info
if exist build rmdir/q/s build

popd
