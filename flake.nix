{
  inputs = {};
  outputs = {nixpkgs, ...}: 
  let devShell = system:
     let pkgs = import nixpkgs { inherit system; };
     in
     pkgs.mkShell {
      buildInputs = with pkgs; [
        python3
        portaudio
      ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (with pkgs; [ portaudio stdenv.cc.cc.lib zlib ]);
    };
  in
  {
    devShells."aarch64-linux".default = devShell "aarch64-linux";
    devShells."aarch64-darwin".default = devShell "aarch64-darwin";
    devShells."x86_64-linux".default = devShell "x86_64-linux";
    devShells."x86_64-darwin".default = devShell "x86_64-darwin";
  };
}
