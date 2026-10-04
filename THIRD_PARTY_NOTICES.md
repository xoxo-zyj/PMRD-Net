# Third-party code and licensing

- `future_process/efficientvit/` contains bundled EfficientViT-derived code.
  Upstream: https://github.com/mit-han-lab/efficientvit. Its Apache-2.0
  license is preserved in `LICENSES/EfficientViT-Apache-2.0.txt`.
- `future_process/vmamba.py` identifies MzeroMiko as its upstream author and
  includes local modifications already present in the supplied archive.
  Upstream: https://github.com/MzeroMiko/VMamba. Its MIT license is preserved
  in `LICENSES/VMamba-MIT.txt`.
- The external selective-scan CUDA extension is built from the VMamba
  project; retain its own notices when distributing that compiled package.

The source archive contained no project-wide license for the author's
original MSDSPDD implementation or checkpoints. This preparation does not
grant a new project-wide license. The maintainer should add their chosen
license before advertising the repository as open-source licensed.
