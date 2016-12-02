#!/bin/sh -x

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

cd "/home/glanger1/catkin_ws/neuroMobile/src"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
/usr/bin/env \
    PYTHONPATH="/home/glanger1/catkin_ws/neuroMobile/install/lib/python2.7/dist-packages:/home/glanger1/catkin_ws/neuroMobile/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/glanger1/catkin_ws/neuroMobile/build" \
    "/usr/bin/python" \
    "/home/glanger1/catkin_ws/neuroMobile/src/setup.py" \
    build --build-base "/home/glanger1/catkin_ws/neuroMobile/build" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/glanger1/catkin_ws/neuroMobile/install" --install-scripts="/home/glanger1/catkin_ws/neuroMobile/install/bin"
