- **One model provider interface** (E6/#1477): `pinneapple_neural.model_provider.load(ref)` opens a PyTorch
  checkpoint, a TorchScript archive, an ONNX graph, a JAX pytree, a PhysicsNeMo `.mdlus` checkpoint or a Hugging Face
  repo through the same call, and returns a model with its provenance (provider, version, reference, SHA-256).
  `export("onnx" | "torchscript" | "artifact")` **refuses with the reason** when the file would compute something other
  than the model — data-dependent control flow, an unsupported operator, or a graph that disagrees with the model on
  the example input — instead of writing it.
