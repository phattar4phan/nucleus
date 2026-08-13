{
  description = "3.13";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        python313

        cudaPackages.cudatoolkit
        cudaPackages.cudnn

        stdenv.cc.cc
        zlib
        openssl
        libffi
      ];

      shellHook = ''
        export UV_PROJECT_ENVIRONMENT=env

        export CUDA_HOME=${pkgs.cudaPackages.cudatoolkit}
        export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}

        export LD_LIBRARY_PATH="${
          pkgs.lib.makeLibraryPath [
            pkgs.cudaPackages.cudatoolkit
            pkgs.cudaPackages.cudnn
            pkgs.stdenv.cc.cc
            pkgs.zlib
          ]
        }:/run/opengl-driver/lib:LD_LIBRARY_PATH

        if [ ! -d env ]; then
          uv venv --python ${pkgs.python313}/bin/python env
        fi

        source env/bin/activate
      '';
    };
  };
}