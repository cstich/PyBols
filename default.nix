with import <nixpkgs> {};

let
  my-python-packages = python-packages: with python-packages; [
    build
    flake8
    jupyter numba
    numpy
    pip
    scipy
    scikit-learn
    statsmodels
    tabulate
    twine
    wheel
  ]; 
  python-with-my-packages = python3.withPackages my-python-packages;
in pkgs.mkShell {
  buildInputs = [
    python-with-my-packages   
  ];
  shellHook = ''
    # Tells pip to put packages into $PIP_PREFIX instead of the usual locations.
    # See https://pip.pypa.io/en/stable/user_guide/#environment-variables.
    export PIP_PREFIX=$(pwd)/_build/pip_packages
    export PYTHONPATH="$PIP_PREFIX/${pkgs.python3.sitePackages}:$PYTHONPATH"
    export PATH="$PIP_PREFIX/bin:$PATH"
    unset SOURCE_DATE_EPOCH
  '';
}

