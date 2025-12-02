// app/page.tsx
"use client";

import { useCallback, useState } from "react";
import * as ort from "onnxruntime-web";

// ---- Minimal NPY parser for float32, C-order arrays ----

function parseNpyFloat32(
  buffer: ArrayBuffer
): { data: Float32Array; shape: number[] } {
  const magic = new Uint8Array(buffer, 0, 6);
  const expectedMagic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY
  for (let i = 0; i < expectedMagic.length; i++) {
    if (magic[i] !== expectedMagic[i]) {
      throw new Error("Invalid NPY file: bad magic string");
    }
  }

  const view = new DataView(buffer);
  const major = view.getUint8(6);
  const minor = view.getUint8(7);

  let headerLen: number;
  let headerStart: number;

  if (major === 1) {
    headerLen = view.getUint16(8, true);
    headerStart = 10;
  } else if (major === 2) {
    headerLen = view.getUint32(8, true);
    headerStart = 12;
  } else {
    throw new Error(`Unsupported NPY version ${major}.${minor}`);
  }

  const headerBytes = new Uint8Array(buffer, headerStart, headerLen);
  const header = new TextDecoder("latin1").decode(headerBytes);

  // Very simple parser: look for "shape": (..)
  const shapeMatch = header.match(/'shape': *\(([^)]*)\)/);
  if (!shapeMatch) {
    throw new Error("Cannot parse NPY header shape");
  }
  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => Number(s));

  if (shape.some((d) => !Number.isFinite(d))) {
    throw new Error(`Invalid shape parsed from NPY header: ${shapeMatch[1]}`);
  }

  // Assume float32, little-endian, C-order
  const dataOffset = headerStart + headerLen;
  const data = new Float32Array(buffer, dataOffset);

  return { data, shape };
}

export default function Home() {
  const [status, setStatus] = useState<string>("");
  const [inferenceTime, setInferenceTime] = useState<string>("");
  const [outputSummary, setOutputSummary] = useState<string>("");

  const runPangu = useCallback(async () => {
    try {
      if (typeof window === "undefined") return;

      if (!("gpu" in navigator)) {
        setStatus("WebGPU not supported in this browser.");
        return;
      }

      setStatus("Creating WebGPU session...");
      setInferenceTime("");
      setOutputSummary("");
      ort.env.debug = true;
ort.env.logLevel = "verbose";

      // Create ONNX Runtime session with WebGPU EP
      // const session = await ort.InferenceSession.create(
      //   "/pangu_weather_1.onnx",
      //   {
      //     executionProviders: ["webgpu"],
      //     graphOptimizationLevel: "all",
      //   }
      // );
  ort.env.wasm.numThreads = 1;  // disable threading
  // optionally: ort.env.wasm.simd = true; // keep SIMD
  const session = await ort.InferenceSession.create("/pangu_weather_1.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

      console.log("Pangu session created with inputs:", session.inputNames);
      setStatus("Loading NPY inputs...");

      // Fetch NPY files from /public
      const [surfaceResp, upperResp] = await Promise.all([
        fetch("/weather/input_surface.npy"),
        fetch("/weather/input_upper.npy"),
      ]);

      if (!surfaceResp.ok || !upperResp.ok) {
        throw new Error("Failed to fetch one or both NPY input files");
      }

      const [surfaceBuf, upperBuf] = await Promise.all([
        surfaceResp.arrayBuffer(),
        upperResp.arrayBuffer(),
      ]);

      const surfaceParsed = parseNpyFloat32(surfaceBuf);
      const upperParsed = parseNpyFloat32(upperBuf);

      const surfaceTensor = new ort.Tensor(
        "float32",
        surfaceParsed.data,
        surfaceParsed.shape
      );
      const upperTensor = new ort.Tensor(
        "float32",
        upperParsed.data,
        upperParsed.shape
      );

      // Map tensors to model inputs using inputNames order
      const feeds: Record<string, ort.Tensor> = {};
      if (session.inputNames.length >= 2) {
        feeds[session.inputNames[0]] = surfaceTensor;
        feeds[session.inputNames[1]] = upperTensor;
      } else if (session.inputNames.length === 1) {
        // If model is wrapped to take a single stacked input, user can adjust this.
        feeds[session.inputNames[0]] = surfaceTensor;
      } else {
        throw new Error("Model has no inputs");
      }

      setStatus("Running inference on WebGPU...");

      const t0 = performance.now();
      const outputMap = await session.run(feeds);
      const t1 = performance.now();

      const dt = (t1 - t0) / 1000;
      setInferenceTime(`${dt.toFixed(3)} s`);

      const outputs = session.outputNames.map((name) => {
        const out = outputMap[name];
        const shape = out.dims.join(" × ");
        return `${name}: [${shape}]`;
      });

      setOutputSummary(outputs.join("\n"));
      setStatus("Done.");
    } catch (err) {
      console.error(err);
      setStatus(
        err instanceof Error ? `Error: ${err.message}` : "Unknown error"
      );
    }
  }, []);

  return (
    <main className="min-h-screen bg-zinc-50 px-6 py-10 text-zinc-900 dark:bg-black dark:text-zinc-50">
      <div className="mx-auto flex max-w-3xl flex-col gap-6">
        <header>
          <h1 className="text-2xl font-semibold">
            Pangu Weather (ONNX) · WebGPU Demo
          </h1>
          <p className="mt-2 text-sm text-zinc-600 dark:text-zinc-400">
            Model file: <code>/pangu_weather_1.onnx</code>, inputs from{" "}
            <code>/weather/input_surface.npy</code> and{" "}
            <code>/weather/input_upper.npy</code>.
          </p>
        </header>

        <button
          onClick={runPangu}
          className="inline-flex w-fit items-center justify-center rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-medium shadow-sm hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800"
        >
          Run Pangu inference (WebGPU)
        </button>

        <section className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span className="whitespace-pre-wrap">
              {status || "Idle. Click the button to run."}
            </span>
          </div>
          {inferenceTime && (
            <div>
              <span className="font-medium">Inference time:</span>{" "}
              {inferenceTime}
            </div>
          )}
          {outputSummary && (
            <div className="mt-2">
              <div className="font-medium">Outputs:</div>
              <pre className="mt-1 rounded-md border border-zinc-200 bg-white p-2 text-xs text-zinc-900 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-100 whitespace-pre-wrap">
                {outputSummary}
              </pre>
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
